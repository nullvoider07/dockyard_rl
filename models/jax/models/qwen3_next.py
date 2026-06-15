"""Qwen3-Next hybrid model (linear + full attention, sparse MoE), Flax NNX (J11c).

JAX mirror of `transformers.models.qwen3_next` (Qwen3.5 lineage). Each decoder
layer is either a linear-attention (Gated-DeltaNet, J11b) or a full-attention
mixer, per `config.layer_types`, followed by a sparse MoE block (with a gated
shared expert, reusing the J8 MoE substrate) or a dense MLP. Distinctive vs the
dense Qwen3: `(1+weight)` RMSNorm, partial RoPE (factor 0.25), and gated
attention (a sigmoid gate on the attention output, packed into a double-width
`q_proj`).

Incremental-decode cache is omitted (the trainer only needs the full-sequence
prefill; KV/conv/recurrent caching is the inference fleet's job). Shape suffixes:
B batch, S seq, D model dim, H heads, Dh head dim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from dockyard_rl.models.jax.layers import Embedding, Linear
from dockyard_rl.models.jax.linear_attn.gated_deltanet import Qwen3NextGatedDeltaNet
from dockyard_rl.models.jax.models.qwen3 import _causal_bias, _rotate_half
from dockyard_rl.models.jax.moe.block import MoEBlock
from dockyard_rl.models.jax.moe.dispatch import LocalTokenDispatcher
from dockyard_rl.models.jax.moe.experts import GroupedExperts
from dockyard_rl.models.jax.moe.router import TokenChoiceTopKRouter

Array = jax.Array


@dataclass(frozen=True)
class Qwen3NextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    tie_word_embeddings: bool
    # linear-attention (Gated-DeltaNet) dims
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    # MoE
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    norm_topk_prob: bool
    decoder_sparse_step: int
    mlp_only_layers: tuple[int, ...]
    layer_types: tuple[str, ...]

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    def is_sparse_layer(self, layer_idx: int) -> bool:
        return (layer_idx not in self.mlp_only_layers) and (
            self.num_experts > 0 and (layer_idx + 1) % self.decoder_sparse_step == 0
        )

    @classmethod
    def from_hf_config(cls, hf: Any) -> "Qwen3NextConfig":
        rp = hf.rope_parameters or {}
        layer_types = tuple(hf.layer_types)
        return cls(
            vocab_size=int(hf.vocab_size), hidden_size=int(hf.hidden_size),
            intermediate_size=int(hf.intermediate_size),
            num_hidden_layers=int(hf.num_hidden_layers),
            num_attention_heads=int(hf.num_attention_heads),
            num_key_value_heads=int(hf.num_key_value_heads), head_dim=int(hf.head_dim),
            rms_norm_eps=float(hf.rms_norm_eps), rope_theta=float(rp.get("rope_theta", 10000.0)),
            partial_rotary_factor=float(rp.get("partial_rotary_factor", 1.0)),
            tie_word_embeddings=bool(hf.tie_word_embeddings),
            linear_num_value_heads=int(hf.linear_num_value_heads),
            linear_num_key_heads=int(hf.linear_num_key_heads),
            linear_key_head_dim=int(hf.linear_key_head_dim),
            linear_value_head_dim=int(hf.linear_value_head_dim),
            linear_conv_kernel_dim=int(hf.linear_conv_kernel_dim),
            num_experts=int(hf.num_experts), num_experts_per_tok=int(hf.num_experts_per_tok),
            moe_intermediate_size=int(hf.moe_intermediate_size),
            shared_expert_intermediate_size=int(hf.shared_expert_intermediate_size),
            norm_topk_prob=bool(hf.norm_topk_prob),
            decoder_sparse_step=int(hf.decoder_sparse_step),
            mlp_only_layers=tuple(hf.mlp_only_layers or ()), layer_types=layer_types,
        )


class RMSNorm1p(nnx.Module):
    """`Qwen3NextRMSNorm`: RMSNorm with a `(1 + weight)` zero-init scale.

    Distinct from the dense Qwen3 `RMSNorm` (ones-init `weight * x`): here the
    scale stores a zero-centered delta and the model normalizes as
    `(1 + weight) * x_normed` (the HF `output * (1.0 + weight.float())` form).
    """

    def __init__(self, dim: int, *, eps: float = 1e-6, param_dtype: Any = jnp.float32) -> None:
        self.dim = dim
        self.eps = eps
        self.scale = nnx.Param(jnp.zeros((dim,), dtype=param_dtype))

    def __call__(self, x: Array) -> Array:
        in_dtype = x.dtype
        xf = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
        normed = xf * jax.lax.rsqrt(variance + self.eps)
        out = normed * (1.0 + self.scale[...].astype(jnp.float32))
        return out.astype(in_dtype)


def _partial_rope_cos_sin(position_ids: Array, rotary_dim: int, theta: float) -> tuple[Array, Array]:
    inv_freq = 1.0 / (theta ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))
    freqs = position_ids[:, :, None].astype(jnp.float32) * inv_freq[None, None, :]
    emb = jnp.concatenate((freqs, freqs), axis=-1)  # (b, s, rotary_dim)
    return jnp.cos(emb), jnp.sin(emb)


def _apply_partial_rope(t: Array, cos: Array, sin: Array) -> Array:
    """Rotate the first `rotary_dim` head dims, pass the remainder through."""
    rd = cos.shape[-1]
    t_rot, t_pass = t[..., :rd], t[..., rd:]
    c = cos[:, :, None, :]
    s = sin[:, :, None, :]
    rotated = t_rot * c + _rotate_half(t_rot) * s
    return jnp.concatenate([rotated, t_pass], axis=-1)


class Qwen3NextAttention(nnx.Module):
    """Gated full attention: double-width `q_proj` (query + sigmoid output gate),
    head-dim QK-norm, partial RoPE, GQA."""

    def __init__(self, cfg: Qwen3NextConfig, *, rngs: nnx.Rngs, param_dtype: Any) -> None:
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.num_kv_groups = cfg.num_attention_heads // cfg.num_key_value_heads
        self.scaling = cfg.head_dim ** -0.5
        h = cfg.hidden_size
        self.q_proj = Linear(h, self.num_heads * self.head_dim * 2, rngs=rngs, param_dtype=param_dtype)
        self.k_proj = Linear(h, self.num_kv_heads * self.head_dim, rngs=rngs, param_dtype=param_dtype)
        self.v_proj = Linear(h, self.num_kv_heads * self.head_dim, rngs=rngs, param_dtype=param_dtype)
        self.o_proj = Linear(self.num_heads * self.head_dim, h, rngs=rngs, param_dtype=param_dtype)
        self.q_norm = RMSNorm1p(self.head_dim, eps=cfg.rms_norm_eps, param_dtype=param_dtype)
        self.k_norm = RMSNorm1p(self.head_dim, eps=cfg.rms_norm_eps, param_dtype=param_dtype)

    def __call__(self, x: Array, cos: Array, sin: Array, attn_bias: Array) -> Array:
        b, s, _ = x.shape
        qg = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim * 2)
        q = self.q_norm(qg[..., : self.head_dim])
        gate = qg[..., self.head_dim :].reshape(b, s, self.num_heads * self.head_dim)
        k = self.k_norm(self.k_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim)

        q = _apply_partial_rope(q, cos, sin)
        k = _apply_partial_rope(k, cos, sin)
        if self.num_kv_groups > 1:
            k = jnp.repeat(k, self.num_kv_groups, axis=2)
            v = jnp.repeat(v, self.num_kv_groups, axis=2)

        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * self.scaling + attn_bias
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
        out = jnp.einsum("bhqk,bkhd->bqhd", probs, v).reshape(b, s, self.num_heads * self.head_dim)
        out = out * jax.nn.sigmoid(gate)
        return self.o_proj(out)


class Qwen3NextMLP(nnx.Module):
    def __init__(self, cfg: Qwen3NextConfig, intermediate_size: int, *, rngs: nnx.Rngs, param_dtype: Any) -> None:
        h = cfg.hidden_size
        self.gate_proj = Linear(h, intermediate_size, rngs=rngs, param_dtype=param_dtype)
        self.up_proj = Linear(h, intermediate_size, rngs=rngs, param_dtype=param_dtype)
        self.down_proj = Linear(intermediate_size, h, rngs=rngs, param_dtype=param_dtype)

    def __call__(self, x: Array) -> Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3NextSharedExpert(nnx.Module):
    """Always-on expert with a sigmoid gate: `sigmoid(gate(x)) * MLP(x)`."""

    def __init__(self, cfg: Qwen3NextConfig, *, rngs: nnx.Rngs, param_dtype: Any) -> None:
        self.mlp = Qwen3NextMLP(cfg, cfg.shared_expert_intermediate_size, rngs=rngs, param_dtype=param_dtype)
        self.shared_expert_gate = Linear(cfg.hidden_size, 1, rngs=rngs, param_dtype=param_dtype)

    def __call__(self, x: Array) -> Array:
        return jax.nn.sigmoid(self.shared_expert_gate(x)) * self.mlp(x)


def build_qwen3next_moe_block(cfg: Qwen3NextConfig, *, rngs: nnx.Rngs, param_dtype: Any) -> MoEBlock:
    """Sparse MoE block with a gated shared expert (reuses the J8 substrate).

    HF Qwen3-Next routes via softmax top-k with `norm_topk_prob` and applies the
    gating weight to the expert OUTPUT (`score_before_experts=False`); experts use
    the fused `gate_up_proj`/`down_proj` layout the J8 `GroupedExperts` mirrors.
    No aux-loss-free bias (Qwen3-Next uses a router aux *loss*, not the bias method;
    irrelevant to the forward logits) -> `load_balance_coeff=None`.
    """
    router = TokenChoiceTopKRouter(
        cfg.hidden_size, cfg.num_experts, top_k=cfg.num_experts_per_tok,
        score_func="softmax", route_norm=cfg.norm_topk_prob, route_scale=1.0,
        rngs=rngs, param_dtype=param_dtype,
    )
    dispatcher = LocalTokenDispatcher(cfg.num_experts, cfg.num_experts_per_tok, score_before_experts=False)
    experts = GroupedExperts(
        cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts, dispatcher,
        rngs=rngs, param_dtype=param_dtype,
    )
    shared = Qwen3NextSharedExpert(cfg, rngs=rngs, param_dtype=param_dtype)
    return MoEBlock(router, experts, num_experts=cfg.num_experts, shared_experts=shared, load_balance_coeff=None)


class Qwen3NextDecoderLayer(nnx.Module):
    def __init__(self, cfg: Qwen3NextConfig, layer_idx: int, *, rngs: nnx.Rngs, param_dtype: Any) -> None:
        self.layer_type = cfg.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(cfg, rngs=rngs, param_dtype=param_dtype)
        else:
            self.self_attn = Qwen3NextAttention(cfg, rngs=rngs, param_dtype=param_dtype)
        if cfg.is_sparse_layer(layer_idx):
            self.mlp: nnx.Module = build_qwen3next_moe_block(cfg, rngs=rngs, param_dtype=param_dtype)
        else:
            self.mlp = Qwen3NextMLP(cfg, cfg.intermediate_size, rngs=rngs, param_dtype=param_dtype)
        self.input_layernorm = RMSNorm1p(cfg.hidden_size, eps=cfg.rms_norm_eps, param_dtype=param_dtype)
        self.post_attention_layernorm = RMSNorm1p(cfg.hidden_size, eps=cfg.rms_norm_eps, param_dtype=param_dtype)

    def __call__(self, x: Array, cos: Array, sin: Array, attn_bias: Array) -> Array:
        residual = x
        h = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            h = self.linear_attn(h)
        else:
            h = self.self_attn(h, cos, sin, attn_bias)
        x = residual + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3NextModel(nnx.Module):
    def __init__(self, cfg: Qwen3NextConfig, *, rngs: nnx.Rngs, param_dtype: Any = jnp.float32) -> None:
        self.cfg = cfg
        self.embed_tokens = Embedding(cfg.vocab_size, cfg.hidden_size, rngs=rngs, param_dtype=param_dtype)
        self.layers = nnx.List(
            [Qwen3NextDecoderLayer(cfg, i, rngs=rngs, param_dtype=param_dtype) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm1p(cfg.hidden_size, eps=cfg.rms_norm_eps, param_dtype=param_dtype)

    def __call__(self, input_ids: Array, position_ids: Optional[Array] = None) -> Array:
        b, s = input_ids.shape
        if position_ids is None:
            position_ids = jnp.arange(s, dtype=jnp.int32)[None, :].repeat(b, axis=0)
        cos, sin = _partial_rope_cos_sin(position_ids, self.cfg.rotary_dim, self.cfg.rope_theta)
        attn_bias = _causal_bias(s, jnp.float32)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, attn_bias)
        return self.norm(x)


class Qwen3NextForCausalLM(nnx.Module):
    def __init__(self, cfg: Qwen3NextConfig, *, rngs: nnx.Rngs, param_dtype: Any = jnp.float32) -> None:
        self.cfg = cfg
        self.model = Qwen3NextModel(cfg, rngs=rngs, param_dtype=param_dtype)
        if not cfg.tie_word_embeddings:
            self.lm_head = Linear(cfg.hidden_size, cfg.vocab_size, rngs=rngs, param_dtype=param_dtype)

    def __call__(self, input_ids: Array, position_ids: Optional[Array] = None) -> Array:
        hidden = self.model(input_ids, position_ids)
        if self.cfg.tie_word_embeddings:
            return jnp.einsum("...h,vh->...v", hidden, self.model.embed_tokens.embedding[...])
        return self.lm_head(hidden)
