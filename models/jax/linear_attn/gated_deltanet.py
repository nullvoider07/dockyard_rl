"""Qwen3NextGatedDeltaNet — the linear-attention token mixer, Flax NNX (J11b).

JAX mirror of `transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextGatedDeltaNet`
(training/prefill path, no decode cache). Input hidden -> two projections
(`in_proj_qkvz`, `in_proj_ba`) -> split into q/k/v/z/b/a -> depthwise causal
short-conv on the concatenated q/k/v -> per-head gates (`beta=sigmoid(b)`,
`g=-exp(A_log)*softplus(a+dt_bias)`) -> GQA repeat of q/k -> gated delta-rule
recurrence (J11a) -> gated RMSNorm with the z gate -> out projection.

Shape suffixes: B batch, S seq, D model dim, Hk key heads, Hv value heads,
Dk key-head dim, Dv value-head dim.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from dockyard_rl.models.jax.layers import Linear
from dockyard_rl.models.jax.linear_attn.delta_rule import causal_conv1d, recurrent_gated_delta_rule

Array = jax.Array


class RMSNormGated(nnx.Module):
    """`Qwen3NextRMSNormGated`: RMSNorm (fp32) then a SiLU(gate) multiplicative gate.

    Matches HF ordering exactly: normalize in fp32, scale by `weight` (init ones),
    then multiply by `silu(gate)` (gate cast to fp32), cast back to input dtype.
    Distinct from the model's `(1+weight)` RMSNorm — this gated norm uses a
    plain ones-initialized weight.
    """

    def __init__(self, dim: int, *, eps: float = 1e-6, param_dtype: Any = jnp.float32) -> None:
        self.weight = nnx.Param(jnp.ones((dim,), dtype=param_dtype))
        self.eps = eps

    def __call__(self, x: Array, gate: Array) -> Array:
        in_dtype = x.dtype
        xf = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
        normed = xf * jax.lax.rsqrt(variance + self.eps)
        h = self.weight[...] * normed.astype(in_dtype)
        h = h.astype(jnp.float32) * jax.nn.silu(gate.astype(jnp.float32))
        return h.astype(in_dtype)


class Qwen3NextGatedDeltaNet(nnx.Module):
    def __init__(self, cfg: Any, *, rngs: nnx.Rngs, param_dtype: Any = jnp.float32) -> None:
        self.num_v_heads = cfg.linear_num_value_heads
        self.num_k_heads = cfg.linear_num_key_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = cfg.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        proj_qkvz = self.key_dim * 2 + self.value_dim * 2
        proj_ba = self.num_v_heads * 2
        self.in_proj_qkvz = Linear(cfg.hidden_size, proj_qkvz, rngs=rngs, param_dtype=param_dtype)
        self.in_proj_ba = Linear(cfg.hidden_size, proj_ba, rngs=rngs, param_dtype=param_dtype)
        # Depthwise causal conv1d (bias=False); HF weight (C,1,K) squeezed to (C,K).
        self.conv_weight = nnx.Param(jnp.zeros((self.conv_dim, self.conv_kernel_size), param_dtype))
        self.dt_bias = nnx.Param(jnp.ones((self.num_v_heads,), param_dtype))
        self.A_log = nnx.Param(jnp.zeros((self.num_v_heads,), param_dtype))
        self.norm = RMSNormGated(self.head_v_dim, eps=cfg.rms_norm_eps, param_dtype=param_dtype)
        self.out_proj = Linear(self.value_dim, cfg.hidden_size, rngs=rngs, param_dtype=param_dtype)

    def _fix_ordering(self, qkvz: Array, ba: Array, b: int, s: int):
        """Split the fused qkvz / ba projections into q,k,v,z,b,a (HF layout)."""
        nk, nv, dk, dv = self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim
        r = nv // nk
        qkvz = qkvz.reshape(b, s, nk, 2 * dk + 2 * dv * r)
        ba = ba.reshape(b, s, nk, 2 * r)
        i1, i2, i3 = dk, 2 * dk, 2 * dk + r * dv
        query = qkvz[..., :i1]
        key = qkvz[..., i1:i2]
        value = qkvz[..., i2:i3].reshape(b, s, nv, dv)
        z = qkvz[..., i3:].reshape(b, s, nv, dv)
        bb = ba[..., :r].reshape(b, s, nv)
        aa = ba[..., r:].reshape(b, s, nv)
        return query, key, value, z, bb, aa

    def __call__(self, hidden_BSD: Array) -> Array:
        b, s, _ = hidden_BSD.shape
        qkvz = self.in_proj_qkvz(hidden_BSD)
        ba = self.in_proj_ba(hidden_BSD)
        query, key, value, z, bb, aa = self._fix_ordering(qkvz, ba, b, s)

        mixed = jnp.concatenate(
            [query.reshape(b, s, -1), key.reshape(b, s, -1), value.reshape(b, s, -1)], axis=-1
        )
        mixed = causal_conv1d(mixed, self.conv_weight[...], None)
        query = mixed[..., : self.key_dim].reshape(b, s, self.num_k_heads, self.head_k_dim)
        key = mixed[..., self.key_dim : 2 * self.key_dim].reshape(b, s, self.num_k_heads, self.head_k_dim)
        value = mixed[..., 2 * self.key_dim :].reshape(b, s, self.num_v_heads, self.head_v_dim)

        beta = jax.nn.sigmoid(bb)
        g = -jnp.exp(self.A_log[...].astype(jnp.float32)) * jax.nn.softplus(
            aa.astype(jnp.float32) + self.dt_bias[...].astype(jnp.float32)
        )
        ratio = self.num_v_heads // self.num_k_heads
        if ratio > 1:
            query = jnp.repeat(query, ratio, axis=2)
            key = jnp.repeat(key, ratio, axis=2)

        core = recurrent_gated_delta_rule(query, key, value, g, beta, use_qk_l2norm=True)
        core = self.norm(core, z)
        core = core.reshape(b, s, self.value_dim)
        return self.out_proj(core)
