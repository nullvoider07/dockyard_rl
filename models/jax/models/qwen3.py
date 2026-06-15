"""Qwen3 dense model in Flax NNX.

Re-implements the compute path of ``transformers.models.qwen3.modeling_qwen3``
(verified against transformers 5.8.1) numerically: RMSNorm with fp32 variance,
SwiGLU MLP (bias-free), grouped-query attention with the Qwen3-signature
per-head ``q_norm``/``k_norm`` applied post-reshape pre-RoPE, default RoPE, and
a pre-norm decoder. Inference KV-cache, sliding-window, and FP8 are out of J1
scope (the trainer always runs full-sequence forwards in bf16/fp32).

The parameter tree mirrors the HF module names so the loader name-map
(``models/jax/weights.py``) is a mechanical path transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from dockyard_rl.models.jax.layers import Embedding, Linear, RMSNorm
from dockyard_rl.models.jax.sharding import AXIS_TP

Array = jax.Array

# Megatron tensor-parallel layout for linear kernels stored (in, out):
#   column-parallel  -> shard the output dim   -> (None, AXIS_TP)
#   row-parallel     -> shard the input dim    -> (AXIS_TP, None)
# Embeddings and norms are replicated at J2 (vocab-parallel embedding deferred).
_COLUMN_PARALLEL = (None, AXIS_TP)
_ROW_PARALLEL = (AXIS_TP, None)


@dataclass(frozen=True)
class Qwen3Config:
    """Dense Qwen3 hyperparameters needed for the forward pass.

    Field names match the HF config; ``from_hf_config`` extracts them from a
    ``transformers`` ``Qwen3Config`` (or any object/dict exposing the same
    attributes).
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    attention_bias: bool = False

    @classmethod
    def from_hf_config(cls, hf: Any) -> "Qwen3Config":
        def get(name: str, default: Any = None) -> Any:
            if isinstance(hf, dict):
                return hf.get(name, default)
            return getattr(hf, name, default)

        head_dim = get("head_dim") or (get("hidden_size") // get("num_attention_heads"))
        rope_params = get("rope_parameters") or {}
        rope_theta = rope_params.get("rope_theta") if isinstance(rope_params, dict) else None
        if rope_theta is None:
            rope_theta = get("rope_theta", 10000.0)
        rope_type = rope_params.get("rope_type", "default") if isinstance(rope_params, dict) else "default"
        if rope_type != "default":
            raise NotImplementedError(
                f"Qwen3 JAX (J1) supports only default RoPE; got rope_type={rope_type!r}. "
                "Scaled-RoPE variants are deferred."
            )
        if bool(get("attention_bias", False)):
            raise NotImplementedError(
                "Qwen3 JAX (J1) supports only bias-free attention (Qwen3 dense default)."
            )
        return cls(
            vocab_size=int(get("vocab_size")),
            hidden_size=int(get("hidden_size")),
            intermediate_size=int(get("intermediate_size")),
            num_hidden_layers=int(get("num_hidden_layers")),
            num_attention_heads=int(get("num_attention_heads")),
            num_key_value_heads=int(get("num_key_value_heads")),
            head_dim=int(head_dim),
            rms_norm_eps=float(get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope_theta),
            tie_word_embeddings=bool(get("tie_word_embeddings", False)),
            attention_bias=False,
        )


def _rotate_half(x: Array) -> Array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate((-x2, x1), axis=-1)


def _apply_rope(t: Array, cos: Array, sin: Array) -> Array:
    # t: (b, s, n_heads, head_dim); cos/sin: (b, s, head_dim) -> broadcast over heads.
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    return t * cos + _rotate_half(t) * sin


def _rope_cos_sin(position_ids: Array, head_dim: int, theta: float) -> tuple[Array, Array]:
    # Default RoPE: inv_freq computed in fp32, attention_scaling == 1.0.
    inv_freq = 1.0 / (
        theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )  # (head_dim/2,)
    freqs = position_ids[:, :, None].astype(jnp.float32) * inv_freq[None, None, :]  # (b, s, hd/2)
    emb = jnp.concatenate((freqs, freqs), axis=-1)  # (b, s, hd)
    return jnp.cos(emb), jnp.sin(emb)


class Qwen3MLP(nnx.Module):
    def __init__(self, cfg: Qwen3Config, *, rngs: nnx.Rngs, param_dtype: jnp.dtype) -> None:
        h, i = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = Linear(h, i, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL)
        self.up_proj = Linear(h, i, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL)
        self.down_proj = Linear(i, h, rngs=rngs, param_dtype=param_dtype, sharding=_ROW_PARALLEL)

    def __call__(self, x: Array) -> Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nnx.Module):
    def __init__(self, cfg: Qwen3Config, *, rngs: nnx.Rngs, param_dtype: jnp.dtype) -> None:
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.num_kv_groups = cfg.num_attention_heads // cfg.num_key_value_heads
        self.scaling = cfg.head_dim ** -0.5

        h = cfg.hidden_size
        self.q_proj = Linear(h, self.num_heads * self.head_dim, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL)
        self.k_proj = Linear(h, self.num_kv_heads * self.head_dim, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL)
        self.v_proj = Linear(h, self.num_kv_heads * self.head_dim, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL)
        self.o_proj = Linear(self.num_heads * self.head_dim, h, rngs=rngs, param_dtype=param_dtype, sharding=_ROW_PARALLEL)
        # q_norm/k_norm operate over head_dim only (Qwen3 signature).
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps, param_dtype=param_dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps, param_dtype=param_dtype)

    def __call__(self, x: Array, cos: Array, sin: Array, attn_bias: Array) -> Array:
        b, s, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if self.num_kv_groups > 1:
            k = jnp.repeat(k, self.num_kv_groups, axis=2)
            v = jnp.repeat(v, self.num_kv_groups, axis=2)

        # scores: (b, h, q, k)
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * self.scaling
        scores = scores + attn_bias  # (1, 1, q, k) additive causal mask
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
        out = jnp.einsum("bhqk,bkhd->bqhd", probs, v)  # (b, s, h, d)
        out = out.reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Qwen3DecoderLayer(nnx.Module):
    def __init__(self, cfg: Qwen3Config, *, rngs: nnx.Rngs, param_dtype: jnp.dtype) -> None:
        self.self_attn = Qwen3Attention(cfg, rngs=rngs, param_dtype=param_dtype)
        self.mlp = Qwen3MLP(cfg, rngs=rngs, param_dtype=param_dtype)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, param_dtype=param_dtype)
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, param_dtype=param_dtype
        )

    def __call__(self, x: Array, cos: Array, sin: Array, attn_bias: Array) -> Array:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attn_bias)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


def _causal_bias(seq_len: int, dtype: jnp.dtype) -> Array:
    # Additive mask (1, 1, q, k): 0 where attendable (k <= q), -inf above.
    i = jnp.arange(seq_len)[:, None]
    j = jnp.arange(seq_len)[None, :]
    allowed = j <= i
    neg_inf = jnp.asarray(jnp.finfo(jnp.float32).min, dtype=jnp.float32)
    bias = jnp.where(allowed, 0.0, neg_inf).astype(dtype)
    return bias[None, None, :, :]


class Qwen3Model(nnx.Module):
    """Embedding + decoder stack + final norm (returns last hidden state)."""

    def __init__(self, cfg: Qwen3Config, *, rngs: nnx.Rngs, param_dtype: jnp.dtype = jnp.float32) -> None:
        self.cfg = cfg
        self.param_dtype = param_dtype
        self.embed_tokens = Embedding(cfg.vocab_size, cfg.hidden_size, rngs=rngs, param_dtype=param_dtype)
        self.layers = nnx.List(
            [Qwen3DecoderLayer(cfg, rngs=rngs, param_dtype=param_dtype) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, param_dtype=param_dtype)

    def __call__(self, input_ids: Array, position_ids: Optional[Array] = None) -> Array:
        b, s = input_ids.shape
        if position_ids is None:
            position_ids = jnp.arange(s, dtype=jnp.int32)[None, :].repeat(b, axis=0)
        x = self.embed_tokens(input_ids)
        cos, sin = _rope_cos_sin(position_ids, self.cfg.head_dim, self.cfg.rope_theta)
        cos = cos.astype(x.dtype)
        sin = sin.astype(x.dtype)
        attn_bias = _causal_bias(s, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, attn_bias)
        return self.norm(x)


class Qwen3ForCausalLM(nnx.Module):
    """Qwen3 dense LM head over the base model.

    When ``tie_word_embeddings`` is set, the LM head shares the embedding matrix
    (no separate ``lm_head`` parameter), so the tie is preserved under training
    — matching HF's ``_tied_weights_keys``. Untied (the Qwen3 dense default)
    keeps a column-parallel ``lm_head``.
    """

    def __init__(self, cfg: Qwen3Config, *, rngs: nnx.Rngs, param_dtype: jnp.dtype = jnp.float32) -> None:
        self.cfg = cfg
        self.tied = cfg.tie_word_embeddings
        self.model = Qwen3Model(cfg, rngs=rngs, param_dtype=param_dtype)
        self.lm_head = (
            None
            if self.tied
            else Linear(
                cfg.hidden_size, cfg.vocab_size, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL
            )
        )

    def __call__(self, input_ids: Array, position_ids: Optional[Array] = None) -> Array:
        hidden = self.model(input_ids, position_ids)
        if self.tied:
            # logits = hidden @ embedding^T (shared param), matching the tied HF head.
            embedding = self.model.embed_tokens.embedding[...]  # (vocab, hidden)
            return jnp.einsum("...h,vh->...v", hidden, embedding)
        assert self.lm_head is not None
        return self.lm_head(hidden)
