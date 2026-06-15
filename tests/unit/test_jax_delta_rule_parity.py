"""J11a: Gated delta-rule core + causal conv1d — CPU parity vs the HF qwen3_next
pure-torch reference.

The JAX `lax.scan` recurrence is checked against BOTH HF torch references — the
per-token `torch_recurrent_gated_delta_rule` (near-exact) and the chunked
`torch_chunk_gated_delta_rule` (the training kernel; identical math, looser tol for
fp32 reassociation) — across S below and above the chunk size, with and without
the in-kernel q/k L2-norm. Causal conv1d is checked against a depthwise `nn.Conv1d`.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

qn = pytest.importorskip("transformers.models.qwen3_next.modeling_qwen3_next")

from dockyard_rl.models.jax.linear_attn.delta_rule import (
    causal_conv1d,
    l2norm,
    recurrent_gated_delta_rule,
)


def _inputs(seed, B=2, S=10, H=3, Dk=8, Dv=8):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, S, H, Dk)).astype(np.float32)
    k = rng.standard_normal((B, S, H, Dk)).astype(np.float32)
    v = rng.standard_normal((B, S, H, Dv)).astype(np.float32)
    # g is the log-decay (<= 0 so exp(g) in (0,1], as in the model: -exp(A_log)*softplus);
    # beta in (0,1) as in the model (sigmoid(b)).
    g = -np.abs(rng.standard_normal((B, S, H)).astype(np.float32)) * 0.5
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((B, S, H)).astype(np.float32)))
    return q, k, v, g, beta


@pytest.mark.parametrize("S", [10, 70])
@pytest.mark.parametrize("use_l2", [False, True])
def test_recurrent_matches_torch_refs(S, use_l2):
    q, k, v, g, beta = _inputs(seed=S + int(use_l2), S=S)
    tq, tk, tv, tg, tb = (torch.from_numpy(x) for x in (q, k, v, g, beta))

    out_recurrent, _ = qn.torch_recurrent_gated_delta_rule(
        tq, tk, tv, g=tg, beta=tb, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=use_l2,
    )
    out_chunk, _ = qn.torch_chunk_gated_delta_rule(
        tq, tk, tv, g=tg, beta=tb, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=use_l2,
    )
    jax_out = np.asarray(
        recurrent_gated_delta_rule(
            jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), jnp.asarray(g), jnp.asarray(beta),
            use_qk_l2norm=use_l2,
        )
    )

    # JAX scan vs the per-token torch recurrence: same algorithm; the only drift is
    # fp32 accumulation order over the sequence (a single ~3e-5 element at S=70).
    np.testing.assert_allclose(jax_out, out_recurrent.numpy(), atol=1e-4, rtol=2e-4)
    # JAX scan vs the chunked training kernel: identical math, fp32 reassociation.
    np.testing.assert_allclose(jax_out, out_chunk.numpy(), atol=2e-4, rtol=2e-4)


def test_l2norm_matches_torch():
    x = np.random.default_rng(0).standard_normal((2, 5, 8)).astype(np.float32)
    got = np.asarray(l2norm(jnp.asarray(x), axis=-1, eps=1e-6))
    ref = qn.l2norm(torch.from_numpy(x), dim=-1, eps=1e-6).numpy()
    np.testing.assert_allclose(got, ref, atol=1e-6, rtol=1e-6)


def test_causal_conv1d_matches_depthwise_conv():
    B, S, C, K = 2, 12, 6, 4
    rng = np.random.default_rng(1)
    x = rng.standard_normal((B, S, C)).astype(np.float32)
    conv = nn.Conv1d(C, C, kernel_size=K, groups=C, padding=K - 1, bias=True)
    assert conv.bias is not None
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(rng.standard_normal(tuple(conv.weight.shape)).astype(np.float32)))
        conv.bias.copy_(torch.from_numpy(rng.standard_normal((C,)).astype(np.float32)))
        # HF prefill path: silu(conv(x_BCS)[..., :S]) then back to (B,S,C)
        ref = F.silu(conv(torch.from_numpy(x).transpose(1, 2))[:, :, :S]).transpose(1, 2).numpy()

    w_CK = conv.weight.detach().squeeze(1).numpy()  # (C, K) == conv1d.weight.squeeze(1)
    got = np.asarray(causal_conv1d(jnp.asarray(x), jnp.asarray(w_CK), jnp.asarray(conv.bias.detach().numpy())))
    np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-5)
