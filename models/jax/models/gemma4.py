"""Gemma 4 (``gemma4_text``) model in Flax NNX.

Re-implements the compute path of ``transformers.models.gemma4`` (verified against
transformers 5.8.1) for the trainer's full-sequence forward. Gemma 4 diverges from
Qwen3 in several ways that are built up sub-phase by sub-phase (see
``handoff``/plan C1-C6):

  - dual head-dim + dual RoPE per layer type (sliding vs full/global);
  - q/k/v RMSNorm (``v_norm`` scale-free) with attention ``scaling = 1.0``;
  - ``attention_k_eq_v`` (global layers reuse K as V) and KV-layer sharing;
  - per-layer input embeddings (PLE); MoE FFN + optional double-wide MLP;
  - ``gelu_pytorch_tanh`` MLP, ``final_logit_softcapping``, ``sqrt(hidden)``
    embedding scale.

This module is **C1**: the config extraction and the Gemma-specific dense
primitives (RMSNorm, MLP, scaled embedding). Attention / KV-sharing / PLE / MoE /
assembly land in C2-C6. The parameter tree mirrors HF module names so the loader
name-map is a mechanical path transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from dockyard_rl.models.jax.layers import Linear
from dockyard_rl.models.jax.models.qwen3 import _COLUMN_PARALLEL, _ROW_PARALLEL, _apply_rope, _causal_bias
from dockyard_rl.models.jax.moe.alltoall import make_token_dispatcher

Array = jax.Array

# Gemma 4's default sliding:full interleave is 5:1 (one full layer every sixth),
# with the last layer forced to full_attention. Mirrors Gemma4TextConfig.__post_init__.
_SLIDING_WINDOW_PATTERN = 6


def _default_layer_types(num_hidden_layers: int) -> tuple[str, ...]:
    types = [
        "sliding_attention" if bool((i + 1) % _SLIDING_WINDOW_PATTERN) else "full_attention"
        for i in range(num_hidden_layers)
    ]
    if types:
        types[-1] = "full_attention"
    return tuple(types)


@dataclass(frozen=True)
class Gemma4TextConfig:
    """Gemma 4 text hyperparameters needed for the forward pass.

    Field names match the HF ``Gemma4TextConfig``; ``from_hf_config`` extracts them
    from a ``transformers`` config object (post-init'd, so ``layer_types`` /
    ``rope_parameters`` are already resolved) or any dict exposing the same keys.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    hidden_activation: str
    tie_word_embeddings: bool
    attention_bias: bool
    # Attention regimes
    global_head_dim: int
    num_global_key_value_heads: int
    attention_k_eq_v: bool
    num_kv_shared_layers: int
    sliding_window: int
    layer_types: tuple[str, ...]
    use_bidirectional_attention: Optional[str]
    final_logit_softcapping: Optional[float]
    # RoPE (per layer-type)
    sliding_rope_type: str
    sliding_rope_theta: float
    global_rope_type: str
    global_rope_theta: float
    global_partial_rotary_factor: float
    # Per-layer input embeddings (PLE)
    vocab_size_per_layer_input: int
    hidden_size_per_layer_input: int
    # MoE
    enable_moe_block: bool
    num_experts: Optional[int]
    top_k_experts: Optional[int]
    moe_intermediate_size: Optional[int]
    use_double_wide_mlp: bool

    @property
    def embed_scale(self) -> float:
        """Main-embedding normalizer (HF: ``hidden_size ** 0.5``)."""
        return float(self.hidden_size) ** 0.5

    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "sliding_attention"

    def head_dim_for(self, layer_idx: int) -> int:
        """Sliding layers use ``head_dim``; full/global layers use ``global_head_dim``."""
        if not self.is_sliding(layer_idx) and self.global_head_dim:
            return self.global_head_dim
        return self.head_dim

    def is_kv_shared_layer(self, layer_idx: int) -> bool:
        first = self.num_hidden_layers - self.num_kv_shared_layers
        return layer_idx >= first > 0

    def mlp_intermediate_size(self, layer_idx: int) -> int:
        """Double-wide MLP applies only on KV-shared layers (HF ``Gemma4TextMLP``)."""
        wide = self.use_double_wide_mlp and self.is_kv_shared_layer(layer_idx)
        return self.intermediate_size * (2 if wide else 1)

    @classmethod
    def from_hf_config(cls, hf: Any) -> "Gemma4TextConfig":
        def get(name: str, default: Any = None) -> Any:
            if isinstance(hf, dict):
                return hf.get(name, default)
            return getattr(hf, name, default)

        num_layers = int(get("num_hidden_layers"))
        head_dim = int(get("head_dim") or (get("hidden_size") // get("num_attention_heads")))
        num_kv_heads = int(get("num_key_value_heads"))
        num_global_kv = get("num_global_key_value_heads")
        num_global_kv = int(num_global_kv) if num_global_kv is not None else num_kv_heads

        layer_types = get("layer_types")
        layer_types = tuple(layer_types) if layer_types else _default_layer_types(num_layers)

        rope = get("rope_parameters") or {}
        local = rope.get("sliding_attention", {}) if isinstance(rope, dict) else {}
        glob = rope.get("full_attention", {}) if isinstance(rope, dict) else {}

        return cls(
            vocab_size=int(get("vocab_size")),
            hidden_size=int(get("hidden_size")),
            intermediate_size=int(get("intermediate_size")),
            num_hidden_layers=num_layers,
            num_attention_heads=int(get("num_attention_heads")),
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=float(get("rms_norm_eps", 1e-6)),
            hidden_activation=str(get("hidden_activation", "gelu_pytorch_tanh")),
            tie_word_embeddings=bool(get("tie_word_embeddings", True)),
            attention_bias=bool(get("attention_bias", False)),
            global_head_dim=int(get("global_head_dim", 512) or 0),
            num_global_key_value_heads=num_global_kv,
            attention_k_eq_v=bool(get("attention_k_eq_v", False)),
            num_kv_shared_layers=int(get("num_kv_shared_layers", 0)),
            sliding_window=int(get("sliding_window", 512)),
            layer_types=layer_types,
            use_bidirectional_attention=get("use_bidirectional_attention"),
            final_logit_softcapping=(
                float(get("final_logit_softcapping"))
                if get("final_logit_softcapping") is not None
                else None
            ),
            sliding_rope_type=str(local.get("rope_type", "default")),
            sliding_rope_theta=float(local.get("rope_theta", 10_000.0)),
            global_rope_type=str(glob.get("rope_type", "proportional")),
            global_rope_theta=float(glob.get("rope_theta", 1_000_000.0)),
            global_partial_rotary_factor=float(glob.get("partial_rotary_factor", 0.25)),
            vocab_size_per_layer_input=int(get("vocab_size_per_layer_input", 0) or 0),
            hidden_size_per_layer_input=int(get("hidden_size_per_layer_input", 0) or 0),
            enable_moe_block=bool(get("enable_moe_block", False)),
            num_experts=(int(get("num_experts")) if get("num_experts") is not None else None),
            top_k_experts=(int(get("top_k_experts")) if get("top_k_experts") is not None else None),
            moe_intermediate_size=(
                int(get("moe_intermediate_size")) if get("moe_intermediate_size") is not None else None
            ),
            use_double_wide_mlp=bool(get("use_double_wide_mlp", False)),
        )


class Gemma4RMSNorm(nnx.Module):
    """Gemma 4 RMSNorm.

    Differs from ``Qwen3RMSNorm`` in two ways: the scale is a plain learned
    ``weight`` (ones-init, ``* weight`` — NOT ``(1 + weight)``), multiplied in
    float32 *before* the cast back to the input dtype; and the reciprocal-sqrt is
    written as ``pow(mean_sq, -0.5)`` to match HF's deliberate torch/JAX
    compiler-parity choice. ``with_scale=False`` (Gemma 4 ``v_norm``) drops the
    weight entirely.
    """

    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-6,
        with_scale: bool = True,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.dim = dim
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nnx.Param(jnp.ones((dim,), dtype=param_dtype))

    def __call__(self, x: Array) -> Array:
        in_dtype = x.dtype
        xf = x.astype(jnp.float32)
        mean_sq = jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + self.eps
        normed = xf * jnp.power(mean_sq, -0.5)
        if self.with_scale:
            normed = normed * self.weight[...].astype(jnp.float32)
        return normed.astype(in_dtype)


class Gemma4MLP(nnx.Module):
    """Gemma 4 gated MLP: ``down(gelu_tanh(gate(x)) * up(x))``, bias-free.

    ``intermediate_size`` is passed in by the caller so the double-wide variant
    (KV-shared layers when ``use_double_wide_mlp``) is selected via
    ``Gemma4TextConfig.mlp_intermediate_size(layer_idx)``.
    """

    def __init__(
        self,
        cfg: Gemma4TextConfig,
        intermediate_size: int,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype,
    ) -> None:
        h = cfg.hidden_size
        i = intermediate_size
        self.gate_proj = Linear(h, i, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL)
        self.up_proj = Linear(h, i, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL)
        self.down_proj = Linear(i, h, rngs=rngs, param_dtype=param_dtype, sharding=_ROW_PARALLEL)

    def __call__(self, x: Array) -> Array:
        return self.down_proj(jax.nn.gelu(self.gate_proj(x), approximate=True) * self.up_proj(x))


class Gemma4ScaledWordEmbedding(nnx.Module):
    """Token embedding table scaled by ``embed_scale`` (HF ``hidden_size ** 0.5``).

    Param name ``embedding`` mirrors the dockyard ``Embedding`` convention; the
    loader maps HF ``embed_tokens.weight`` onto it. The scale is applied in the
    embedding (param) dtype, matching HF's ``* embed_scale.to(weight.dtype)``.
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        embed_scale: float,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
        sharding: Optional[tuple] = None,
    ) -> None:
        self.num_embeddings = num_embeddings
        self.features = features
        self.embed_scale = float(embed_scale)
        table = nnx.initializers.normal(stddev=1.0)(rngs.params(), (num_embeddings, features), param_dtype)
        self.embedding = nnx.Param(table, logical_axes=sharding)

    def __call__(self, ids: Array) -> Array:
        emb = jnp.take(self.embedding[...], ids, axis=0)
        return emb * jnp.asarray(self.embed_scale, dtype=emb.dtype)


# C2: per-layer-type rotary embeddings + attention.


def gemma4_inv_freq(cfg: Gemma4TextConfig, layer_type: str) -> Array:
    """Inverse frequencies for a Gemma 4 layer type (attention_scaling is 1.0 for both).

    Sliding layers: default full-rotary RoPE over ``head_dim`` at ``sliding_rope_theta``.
    Full/global layers: *proportional partial-rotary* over ``global_head_dim`` at
    ``global_rope_theta`` — only ``rope_angles = int(partial_rotary_factor * head_dim // 2)``
    low frequencies are rotated; the remaining ``head_dim//2 - rope_angles`` are zero
    (identity / NoPE). Mirrors transformers ``compute_default_rope_parameters`` /
    ``_compute_proportional_rope_parameters`` (factor defaults to 1.0).
    """
    if layer_type == "sliding_attention":
        dim = cfg.head_dim
        base = cfg.sliding_rope_theta
        return 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    head_dim = cfg.global_head_dim
    base = cfg.global_rope_theta
    rope_angles = int(cfg.global_partial_rotary_factor * head_dim // 2)
    inv_rot = 1.0 / (base ** (jnp.arange(0, 2 * rope_angles, 2, dtype=jnp.float32) / head_dim))
    nope = head_dim // 2 - rope_angles
    if nope > 0:
        return jnp.concatenate([inv_rot, jnp.zeros((nope,), dtype=jnp.float32)])
    return inv_rot


def gemma4_rope_cos_sin(position_ids: Array, inv_freq: Array) -> tuple[Array, Array]:
    """cos/sin of shape ``(b, s, 2*len(inv_freq))`` for a layer type's ``inv_freq``."""
    freqs = position_ids[:, :, None].astype(jnp.float32) * inv_freq[None, None, :]
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def gemma4_sliding_bias(seq_len: int, sliding_window: int, dtype: jnp.dtype) -> Array:
    """Additive mask (1,1,q,k): attend where ``k <= q`` (causal) and ``q - k <
    sliding_window``. Full layers use ``qwen3._causal_bias``; sliding layers use this.
    """
    i = jnp.arange(seq_len)[:, None]
    j = jnp.arange(seq_len)[None, :]
    allowed = (j <= i) & (i - j < sliding_window)
    neg_inf = jnp.asarray(jnp.finfo(jnp.float32).min, dtype=jnp.float32)
    return jnp.where(allowed, 0.0, neg_inf).astype(dtype)[None, None, :, :]


def _store_full_length_kv(cfg: Gemma4TextConfig, layer_idx: int) -> bool:
    """Whether this (non-shared) layer publishes its K/V for the KV-shared tail layers.

    Mirrors HF: among the non-shared layers (``layer_types[:first_shared]``), the
    publisher of a given type is the *last* non-shared layer of that type.
    """
    first_shared = cfg.num_hidden_layers - cfg.num_kv_shared_layers
    if cfg.is_kv_shared_layer(layer_idx):
        return False
    prev = cfg.layer_types[:first_shared]
    lt = cfg.layer_types[layer_idx]
    if lt not in prev:
        return False
    last_of_type = max(i for i in range(len(prev)) if prev[i] == lt)
    return layer_idx == last_of_type


class Gemma4Attention(nnx.Module):
    """Gemma 4 single-layer attention.

    Per-layer ``head_dim`` (sliding -> ``head_dim``, full -> ``global_head_dim``),
    ``scaling = 1.0`` (not ``1/sqrt(d)``), q/k/v RMSNorm with a scale-free ``v_norm``,
    RoPE on q/k only, GQA repeat. Two Gemma-specific modes:

    - ``attention_k_eq_v`` (global layers): no ``v_proj``; V reuses the K projection
      output (``v = v_norm(k_proj(x))``, no RoPE), with ``num_global_key_value_heads``.
    - **KV-layer sharing**: shared tail layers (``is_kv_shared_layer``) hold no
      k/v/k_norm/v_norm weights and consume ``shared_kv`` published by the last
      non-shared layer of their type (``store_full_length_kv``).

    ``__call__`` returns ``(out, published_kv | None)``; the model threads the
    published K/V into the shared layers via a per-layer-type dict.
    """

    def __init__(
        self,
        cfg: Gemma4TextConfig,
        layer_idx: int,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype,
    ) -> None:
        self.head_dim = cfg.head_dim_for(layer_idx)
        self.num_heads = cfg.num_attention_heads
        self.is_kv_shared_layer = cfg.is_kv_shared_layer(layer_idx)
        self.store_full_length_kv = _store_full_length_kv(cfg, layer_idx)
        self.use_k_eq_v = cfg.attention_k_eq_v and not cfg.is_sliding(layer_idx)
        self.num_kv_heads = cfg.num_global_key_value_heads if self.use_k_eq_v else cfg.num_key_value_heads
        self.scaling = 1.0
        h = cfg.hidden_size

        self.q_proj = Linear(h, self.num_heads * self.head_dim, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL)
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=cfg.rms_norm_eps, param_dtype=param_dtype)
        self.o_proj = Linear(self.num_heads * self.head_dim, h, rngs=rngs, param_dtype=param_dtype, sharding=_ROW_PARALLEL)

        # Shared tail layers carry no k/v projections or norms.
        if not self.is_kv_shared_layer:
            self.k_proj = Linear(h, self.num_kv_heads * self.head_dim, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL)
            self.k_norm = Gemma4RMSNorm(self.head_dim, eps=cfg.rms_norm_eps, param_dtype=param_dtype)
            self.v_norm = Gemma4RMSNorm(self.head_dim, eps=cfg.rms_norm_eps, with_scale=False, param_dtype=param_dtype)
            # k_eq_v drops v_proj (V reuses the K projection output).
            self.v_proj = (
                None if self.use_k_eq_v
                else Linear(h, self.num_kv_heads * self.head_dim, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL)
            )

    def __call__(
        self,
        x: Array,
        cos: Array,
        sin: Array,
        attn_bias: Array,
        shared_kv: Optional[tuple[Array, Array]] = None,
    ) -> tuple[Array, Optional[tuple[Array, Array]]]:
        b, s, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim))
        q = _apply_rope(q, cos, sin)

        if self.is_kv_shared_layer:
            assert shared_kv is not None, "KV-shared layer requires shared_kv from its publisher"
            k, v = shared_kv
            published: Optional[tuple[Array, Array]] = None
        else:
            k_raw = self.k_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim)
            v_raw = k_raw if self.use_k_eq_v else self.v_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim)  # type: ignore[union-attr]
            k = _apply_rope(self.k_norm(k_raw), cos, sin)
            v = self.v_norm(v_raw)
            published = (k, v) if self.store_full_length_kv else None

        groups = self.num_heads // k.shape[2]
        if groups > 1:
            k = jnp.repeat(k, groups, axis=2)
            v = jnp.repeat(v, groups, axis=2)

        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * self.scaling
        scores = scores + attn_bias
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
        out = jnp.einsum("bhqk,bkhd->bqhd", probs, v).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out), published


# C4: per-layer input embeddings (PLE).


class Gemma4PerLayerEmbeddings(nnx.Module):
    """Per-Layer Embeddings (PLE).

    Combines a token-identity embedding (``embed_tokens_per_layer``, a packed
    ``[vocab_per_layer, num_layers * ple_dim]`` table scaled by ``sqrt(ple_dim)``)
    with a context projection of ``inputs_embeds`` (``per_layer_model_projection``
    scaled by ``1/sqrt(hidden)``, RMSNorm'd), as
    ``(proj + token_identity) * 2**-0.5``. Mirrors HF
    ``Gemma4TextModel.{get,project}_per_layer_inputs``. Returns
    ``(B, S, num_hidden_layers, ple_dim)``; the model slices ``[:, :, i, :]`` per layer.
    """

    def __init__(self, cfg: Gemma4TextConfig, *, rngs: nnx.Rngs, param_dtype: jnp.dtype) -> None:
        self.cfg = cfg
        ple = cfg.hidden_size_per_layer_input
        self.embed_tokens_per_layer = Gemma4ScaledWordEmbedding(
            cfg.vocab_size_per_layer_input, cfg.num_hidden_layers * ple,
            float(ple) ** 0.5, rngs=rngs, param_dtype=param_dtype,
        )
        self.per_layer_model_projection = Linear(
            cfg.hidden_size, cfg.num_hidden_layers * ple, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL,
        )
        self.per_layer_projection_norm = Gemma4RMSNorm(ple, eps=cfg.rms_norm_eps, param_dtype=param_dtype)
        self.per_layer_model_projection_scale = float(cfg.hidden_size) ** -0.5
        self.per_layer_input_scale = 2.0**-0.5

    def __call__(self, input_ids: Array, inputs_embeds: Array) -> Array:
        cfg = self.cfg
        ple = cfg.hidden_size_per_layer_input
        b, s = input_ids.shape
        token_identity = self.embed_tokens_per_layer(input_ids).reshape(b, s, cfg.num_hidden_layers, ple)
        proj = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        proj = proj.reshape(b, s, cfg.num_hidden_layers, ple)
        proj = self.per_layer_projection_norm(proj)
        return (proj + token_identity) * self.per_layer_input_scale


# C5: MoE (Gemma-specific router + grouped gelu experts).


class Gemma4Router(nnx.Module):
    """Gemma 4 MoE router.

    Scale-free RMSNorm, then ``h * scale * hidden**-0.5``, a bias-free projection
    to ``num_experts``, softmax, top-k, weight-normalize, and a per-expert scale on
    the selected weights. Mirrors HF ``Gemma4TextRouter`` (returns ``(top_k_weights,
    top_k_index)``; the router-probability output is unused downstream).
    """

    def __init__(self, cfg: Gemma4TextConfig, *, rngs: nnx.Rngs, param_dtype: jnp.dtype) -> None:
        d, e = cfg.hidden_size, cfg.num_experts
        assert e is not None and cfg.top_k_experts is not None
        self.top_k = cfg.top_k_experts
        self.scalar_root_size = float(d) ** -0.5
        self.norm = Gemma4RMSNorm(d, eps=cfg.rms_norm_eps, with_scale=False, param_dtype=param_dtype)
        self.proj = Linear(d, e, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL)
        self.scale = nnx.Param(jnp.ones((d,), dtype=param_dtype))
        self.per_expert_scale = nnx.Param(jnp.ones((e,), dtype=param_dtype))

    def __call__(self, h_TD: Array) -> tuple[Array, Array]:
        h = self.norm(h_TD)
        h = h * self.scale[...] * self.scalar_root_size
        probs = jax.nn.softmax(self.proj(h), axis=-1)
        top_w, top_i = jax.lax.top_k(probs, self.top_k)
        top_w = top_w / jnp.sum(top_w, axis=-1, keepdims=True)
        top_w = top_w * self.per_expert_scale[...][top_i]
        return top_w, top_i


class Gemma4Experts(nnx.Module):
    """Grouped routed experts with a fused gate/up projection and gelu-tanh.

    Weights mirror HF ``Gemma4TextExperts``: ``gate_up_proj (E, 2F, D)`` and
    ``down_proj (E, D, F)``. The grouped GEMM runs via ``jax.lax.ragged_dot`` over
    expert-contiguous tokens (CPU-exact), dispatched/combined by a ``local`` token
    dispatcher with ``score_before_experts=False`` so the router weights multiply
    the expert OUTPUT (matching HF). Distinct from ``moe.GroupedExperts`` (SwiGLU/silu).
    """

    def __init__(self, cfg: Gemma4TextConfig, *, rngs: nnx.Rngs, param_dtype: jnp.dtype) -> None:
        e, f, d, k = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size, cfg.top_k_experts
        assert e is not None and f is not None and k is not None
        self.num_experts = e
        k1, k2 = jax.random.split(rngs.params(), 2)
        scale = 1.0 / jnp.sqrt(jnp.asarray(d, jnp.float32))
        dscale = 1.0 / jnp.sqrt(jnp.asarray(f, jnp.float32))
        self.gate_up_proj = nnx.Param(jax.random.normal(k1, (e, 2 * f, d), param_dtype) * scale)
        self.down_proj = nnx.Param(jax.random.normal(k2, (e, d, f), param_dtype) * dscale)
        self.dispatcher = make_token_dispatcher("local", e, k, score_before_experts=False)

    def _forward(self, x_RD: Array, counts_E: Array) -> Array:
        gate_up = jax.lax.ragged_dot(x_RD, jnp.swapaxes(self.gate_up_proj[...], -1, -2), counts_E)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        act = jax.nn.gelu(gate, approximate=True) * up
        return jax.lax.ragged_dot(act, jnp.swapaxes(self.down_proj[...], -1, -2), counts_E)

    def __call__(self, x_BLD: Array, topk_scores_BLK: Array, topk_ids_BLK: Array) -> Array:
        b, l, d = x_BLD.shape
        k = topk_scores_BLK.shape[-1]
        x_TD = x_BLD.reshape(b * l, d)
        routed_RD, counts_E, meta = self.dispatcher.dispatch(
            x_TD, topk_scores_BLK.reshape(b * l, k), topk_ids_BLK.reshape(b * l, k)
        )
        out_TD = self.dispatcher.combine(self._forward(routed_RD, counts_E), meta, x_TD)
        return out_TD.reshape(b, l, d)


# C6: decoder layer, model, causal LM, HF loader.


class Gemma4DecoderLayer(nnx.Module):
    """Gemma 4 decoder layer (sandwich norms; optional additive MoE and PLE).

    Norm placement is Gemma's post-sublayer "sandwich": the attention/FFN output is
    normed *before* the residual add. When ``enable_moe_block``, the dense MLP and a
    routed-MoE branch (router/experts on the pre-FFN residual) are computed and
    **summed**. When PLE is configured, a per-layer gated residual is added last.
    Returns ``(hidden, published_kv)``.
    """

    def __init__(self, cfg: Gemma4TextConfig, layer_idx: int, *, rngs: nnx.Rngs, param_dtype: jnp.dtype) -> None:
        eps = cfg.rms_norm_eps
        h = cfg.hidden_size
        self.self_attn = Gemma4Attention(cfg, layer_idx, rngs=rngs, param_dtype=param_dtype)
        self.mlp = Gemma4MLP(cfg, cfg.mlp_intermediate_size(layer_idx), rngs=rngs, param_dtype=param_dtype)
        self.input_layernorm = Gemma4RMSNorm(h, eps=eps, param_dtype=param_dtype)
        self.post_attention_layernorm = Gemma4RMSNorm(h, eps=eps, param_dtype=param_dtype)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(h, eps=eps, param_dtype=param_dtype)
        self.post_feedforward_layernorm = Gemma4RMSNorm(h, eps=eps, param_dtype=param_dtype)

        self.has_ple = bool(cfg.hidden_size_per_layer_input)
        if self.has_ple:
            ple = cfg.hidden_size_per_layer_input
            self.per_layer_input_gate = Linear(h, ple, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL)
            self.per_layer_projection = Linear(ple, h, rngs=rngs, param_dtype=param_dtype, sharding=_ROW_PARALLEL)
            self.post_per_layer_input_norm = Gemma4RMSNorm(h, eps=eps, param_dtype=param_dtype)

        self.enable_moe = cfg.enable_moe_block
        if self.enable_moe:
            self.router = Gemma4Router(cfg, rngs=rngs, param_dtype=param_dtype)
            self.experts = Gemma4Experts(cfg, rngs=rngs, param_dtype=param_dtype)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(h, eps=eps, param_dtype=param_dtype)
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(h, eps=eps, param_dtype=param_dtype)
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(h, eps=eps, param_dtype=param_dtype)

    def __call__(
        self,
        hidden: Array,
        per_layer_input: Optional[Array],
        shared_kv: Optional[tuple[Array, Array]],
        cos: Array,
        sin: Array,
        attn_bias: Array,
    ) -> tuple[Array, Optional[tuple[Array, Array]]]:
        residual = hidden
        attn_out, published = self.self_attn(self.input_layernorm(hidden), cos, sin, attn_bias, shared_kv)
        hidden = residual + self.post_attention_layernorm(attn_out)

        residual = hidden
        mlp_out = self.mlp(self.pre_feedforward_layernorm(hidden))
        if self.enable_moe:
            b, s, d = residual.shape
            res_flat = residual.reshape(b * s, d)
            top_w, top_i = self.router(res_flat)
            moe_in = self.pre_feedforward_layernorm_2(res_flat).reshape(b, s, d)
            moe_out = self.experts(moe_in, top_w.reshape(b, s, -1), top_i.reshape(b, s, -1))
            hidden = self.post_feedforward_layernorm_1(mlp_out) + self.post_feedforward_layernorm_2(moe_out)
        else:
            hidden = mlp_out
        hidden = residual + self.post_feedforward_layernorm(hidden)

        if self.has_ple:
            assert per_layer_input is not None
            residual = hidden
            g = jax.nn.gelu(self.per_layer_input_gate(hidden), approximate=True) * per_layer_input
            hidden = residual + self.post_per_layer_input_norm(self.per_layer_projection(g))

        return hidden, published


class Gemma4Model(nnx.Module):
    """Embedding (scaled) + PLE + decoder stack (KV-sharing threaded per layer
    type) + final norm. Returns the last hidden state."""

    def __init__(self, cfg: Gemma4TextConfig, *, rngs: nnx.Rngs, param_dtype: jnp.dtype = jnp.float32) -> None:
        self.cfg = cfg
        self.embed_tokens = Gemma4ScaledWordEmbedding(
            cfg.vocab_size, cfg.hidden_size, cfg.embed_scale, rngs=rngs, param_dtype=param_dtype
        )
        self.layers = nnx.List(
            [Gemma4DecoderLayer(cfg, i, rngs=rngs, param_dtype=param_dtype) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, param_dtype=param_dtype)
        self.has_ple = bool(cfg.hidden_size_per_layer_input)
        if self.has_ple:
            self.per_layer = Gemma4PerLayerEmbeddings(cfg, rngs=rngs, param_dtype=param_dtype)

    def __call__(self, input_ids: Array, position_ids: Optional[Array] = None) -> Array:
        cfg = self.cfg
        b, s = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)
        per_layer_inputs = self.per_layer(input_ids, inputs_embeds) if self.has_ple else None

        if position_ids is None:
            position_ids = jnp.arange(s, dtype=jnp.int32)[None, :].repeat(b, axis=0)

        cos_sin = {lt: gemma4_rope_cos_sin(position_ids, gemma4_inv_freq(cfg, lt)) for lt in set(cfg.layer_types)}
        masks = {
            "full_attention": _causal_bias(s, inputs_embeds.dtype),
            "sliding_attention": gemma4_sliding_bias(s, cfg.sliding_window, inputs_embeds.dtype),
        }

        hidden = inputs_embeds
        shared_kv_states: dict[str, tuple[Array, Array]] = {}
        for i, layer in enumerate(self.layers):
            lt = cfg.layer_types[i]
            cos, sin = cos_sin[lt]
            shared = shared_kv_states.get(lt) if layer.self_attn.is_kv_shared_layer else None
            pli = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            hidden, published = layer(hidden, pli, shared, cos.astype(hidden.dtype), sin.astype(hidden.dtype), masks[lt])
            if published is not None:
                shared_kv_states[lt] = published
        return self.norm(hidden)


class Gemma4ForCausalLM(nnx.Module):
    """Gemma 4 LM head over the base model, with ``final_logit_softcapping`` and
    tied/untied word embeddings (HF default: tied)."""

    def __init__(self, cfg: Gemma4TextConfig, *, rngs: nnx.Rngs, param_dtype: jnp.dtype = jnp.float32) -> None:
        self.cfg = cfg
        self.tied = cfg.tie_word_embeddings
        self.model = Gemma4Model(cfg, rngs=rngs, param_dtype=param_dtype)
        self.lm_head = (
            None if self.tied
            else Linear(cfg.hidden_size, cfg.vocab_size, rngs=rngs, param_dtype=param_dtype, sharding=_COLUMN_PARALLEL)
        )

    def __call__(self, input_ids: Array, position_ids: Optional[Array] = None) -> Array:
        hidden = self.model(input_ids, position_ids)
        if self.tied:
            # Tied head uses the raw (unscaled) embedding table.
            logits = jnp.einsum("...h,vh->...v", hidden, self.model.embed_tokens.embedding[...])
        else:
            assert self.lm_head is not None
            logits = self.lm_head(hidden)
        sc = self.cfg.final_logit_softcapping
        if sc is not None:
            logits = sc * jnp.tanh(logits / sc)
        return logits


def load_hf_gemma4_state_dict(
    model: Gemma4ForCausalLM, sd: Any, *, param_dtype: jnp.dtype = jnp.float32
) -> None:
    """Load an HF Gemma4 ``gemma4_text`` state dict into the NNX model.

    HF ``nn.Linear`` weights are ``(out, in)`` and transpose into the ``(in, out)``
    kernel convention; RMSNorm/embedding/expert tensors map by name. Shared-tail
    attention layers and scale-free norms (``v_norm``, router ``norm``) carry no
    weights and are skipped. ``layer_scalar`` (a ones buffer) is identity and ignored.
    """
    def t(name: str) -> Array:
        v = sd[name]
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else v
        return jnp.asarray(arr).astype(param_dtype)

    g = model.model
    g.embed_tokens.embedding[...] = t("model.embed_tokens.weight")
    g.norm.weight[...] = t("model.norm.weight")
    if g.has_ple:
        g.per_layer.embed_tokens_per_layer.embedding[...] = t("model.embed_tokens_per_layer.weight")
        g.per_layer.per_layer_model_projection.kernel[...] = t("model.per_layer_model_projection.weight").T
        g.per_layer.per_layer_projection_norm.weight[...] = t("model.per_layer_projection_norm.weight")

    for i, layer in enumerate(g.layers):
        p = f"model.layers.{i}."
        layer.input_layernorm.weight[...] = t(p + "input_layernorm.weight")
        layer.post_attention_layernorm.weight[...] = t(p + "post_attention_layernorm.weight")
        layer.pre_feedforward_layernorm.weight[...] = t(p + "pre_feedforward_layernorm.weight")
        layer.post_feedforward_layernorm.weight[...] = t(p + "post_feedforward_layernorm.weight")

        a = layer.self_attn
        a.q_proj.kernel[...] = t(p + "self_attn.q_proj.weight").T
        a.o_proj.kernel[...] = t(p + "self_attn.o_proj.weight").T
        a.q_norm.weight[...] = t(p + "self_attn.q_norm.weight")
        if not a.is_kv_shared_layer:
            a.k_proj.kernel[...] = t(p + "self_attn.k_proj.weight").T
            a.k_norm.weight[...] = t(p + "self_attn.k_norm.weight")
            if a.v_proj is not None:
                a.v_proj.kernel[...] = t(p + "self_attn.v_proj.weight").T

        layer.mlp.gate_proj.kernel[...] = t(p + "mlp.gate_proj.weight").T
        layer.mlp.up_proj.kernel[...] = t(p + "mlp.up_proj.weight").T
        layer.mlp.down_proj.kernel[...] = t(p + "mlp.down_proj.weight").T

        if layer.enable_moe:
            layer.router.proj.kernel[...] = t(p + "router.proj.weight").T
            layer.router.scale[...] = t(p + "router.scale")
            layer.router.per_expert_scale[...] = t(p + "router.per_expert_scale")
            layer.experts.gate_up_proj[...] = t(p + "experts.gate_up_proj")
            layer.experts.down_proj[...] = t(p + "experts.down_proj")
            layer.post_feedforward_layernorm_1.weight[...] = t(p + "post_feedforward_layernorm_1.weight")
            layer.post_feedforward_layernorm_2.weight[...] = t(p + "post_feedforward_layernorm_2.weight")
            layer.pre_feedforward_layernorm_2.weight[...] = t(p + "pre_feedforward_layernorm_2.weight")

        if layer.has_ple:
            layer.per_layer_input_gate.kernel[...] = t(p + "per_layer_input_gate.weight").T
            layer.per_layer_projection.kernel[...] = t(p + "per_layer_projection.weight").T
            layer.post_per_layer_input_norm.weight[...] = t(p + "post_per_layer_input_norm.weight")

    if not model.tied:
        model.lm_head.kernel[...] = t("lm_head.weight").T  # type: ignore[union-attr]
