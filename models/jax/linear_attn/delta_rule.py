"""Gated delta-rule linear attention + causal short-conv, pure JAX (J11a).

JAX port of the HF `qwen3_next` reference functions (the pure-torch fallback path
in `modeling_qwen3_next.py`: `torch_recurrent_gated_delta_rule`,
`torch_chunk_gated_delta_rule`, `l2norm`, and the depthwise causal `conv1d`). The
recurrence is implemented as a `jax.lax.scan` over the sequence — the exact,
simplest form of the delta rule; the HF chunked kernel is a perf optimization that
computes identical math (the J11a parity test asserts the scan matches BOTH torch
references). The chunked/parallel form is deferred to a later throughput pass.

Recurrence (per (batch, head), state S in R^{dk x dv}, gate g_t scalar, beta_t scalar):
    S <- exp(g_t) * S
    kv <- sum_dk(S * k_t[:,None])          # (dv,)
    S <- S + k_t[:,None] * ((v_t - kv) * beta_t)[None,:]
    out_t <- sum_dk(S * q_t[:,None])       # (dv,)
with q scaled by 1/sqrt(dk) and an optional L2-norm on q,k.

Shape suffixes: B batch, S seq, H heads, Dk key-head dim, Dv value-head dim,
C conv channels, K conv kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def l2norm(x: Array, axis: int = -1, eps: float = 1e-6) -> Array:
    """L2 normalization matching the FLA `l2norm` (rsqrt of sum-of-squares + eps)."""
    inv = jax.lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv


def causal_conv1d(x_BSC: Array, weight_CK: Array, bias_C: Array | None = None) -> Array:
    """Depthwise causal short-convolution + SiLU (the prefill path).

    Mirrors HF's `F.silu(conv1d(x)[..., :seq_len])` with a depthwise `nn.Conv1d`
    (`groups=channels`, `padding=K-1`): each channel is convolved along time with
    its own `K`-tap filter, left-padded by `K-1` so output `t` sees inputs
    `[t-K+1 .. t]` (causal). ``weight_CK`` is the HF `conv1d.weight.squeeze(1)`
    (cross-correlation, no kernel flip).
    """
    k = weight_CK.shape[-1]
    xpad = jnp.pad(x_BSC, ((0, 0), (k - 1, 0), (0, 0)))  # left-pad time by K-1
    seq = x_BSC.shape[1]
    # out[t] = sum_j xpad[t+j] * weight[:, j]; original index of tap j is t-(K-1)+j.
    acc = jnp.zeros_like(x_BSC)
    for j in range(k):
        acc = acc + xpad[:, j : j + seq, :] * weight_CK[:, j]
    if bias_C is not None:
        acc = acc + bias_C
    return acc * jax.nn.sigmoid(acc)  # SiLU


def recurrent_gated_delta_rule(
    query_BSHk: Array,
    key_BSHk: Array,
    value_BSHv: Array,
    g_BSH: Array,
    beta_BSH: Array,
    use_qk_l2norm: bool = False,
) -> Array:
    """Gated delta-rule linear attention via `lax.scan` over the sequence.

    Inputs match the HF reference layout: q/k `(B, S, H, Dk)`, v `(B, S, H, Dv)`,
    g/beta `(B, S, H)`. `g` is the log-decay (the recurrence applies `exp(g)`).
    Returns the core attention output `(B, S, H, Dv)` in fp32 (the caller casts).
    """
    qf = query_BSHk.astype(jnp.float32)
    kf = key_BSHk.astype(jnp.float32)
    vf = value_BSHv.astype(jnp.float32)
    gf = g_BSH.astype(jnp.float32)
    bf = beta_BSH.astype(jnp.float32)
    if use_qk_l2norm:
        qf = l2norm(qf, axis=-1, eps=1e-6)
        kf = l2norm(kf, axis=-1, eps=1e-6)

    dk = qf.shape[-1]
    qf = qf * (dk ** -0.5)

    # Move time to the leading axis for the scan: (S, B, H, D).
    q_t = jnp.moveaxis(qf, 1, 0)
    k_t = jnp.moveaxis(kf, 1, 0)
    v_t = jnp.moveaxis(vf, 1, 0)
    g_t = jnp.moveaxis(gf, 1, 0)
    b_t = jnp.moveaxis(bf, 1, 0)

    b, h, dv = value_BSHv.shape[0], value_BSHv.shape[2], value_BSHv.shape[-1]
    state0 = jnp.zeros((b, h, dk, dv), jnp.float32)  # S: (B, H, Dk, Dv)

    def step(state: Array, xs: tuple[Array, Array, Array, Array, Array]):
        q_i, k_i, v_i, g_i, beta_i = xs  # q/k (B,H,Dk), v (B,H,Dv), g/beta (B,H)
        state = state * jnp.exp(g_i)[..., None, None]
        kv = jnp.sum(state * k_i[..., None], axis=-2)  # (B,H,Dv)
        delta = (v_i - kv) * beta_i[..., None]
        state = state + k_i[..., None] * delta[..., None, :]
        out = jnp.sum(state * q_i[..., None], axis=-2)  # (B,H,Dv)
        return state, out

    _final, out_SBHv = jax.lax.scan(step, state0, (q_t, k_t, v_t, g_t, b_t))
    return jnp.moveaxis(out_SBHv, 0, 1)  # (B, S, H, Dv)
