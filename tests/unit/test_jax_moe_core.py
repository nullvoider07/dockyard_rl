"""J8a: MoE compute core CPU parity (router / experts / dispatch / block / load-balance).

The torch router, local dispatcher, and load-balance math are pure and run on CPU,
so they are direct parity references. The torch GroupedExperts use ``torch._grouped_mm``
(CUDA-only), so the JAX ``ragged_dot`` experts and the block are checked against an
explicit per-expert SwiGLU reference instead. Top-k selection order is unspecified,
so comparisons use order-invariant quantities (full scores, routing maps, final outputs).
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
nnx = pytest.importorskip("flax.nnx")

from dockyard_rl.models.dtensor.moe.dispatch import LocalTokenDispatcher as TorchDispatcher
from dockyard_rl.models.dtensor.moe.load_balance import (
    compute_expert_bias_delta as torch_bias_delta,
)
from dockyard_rl.models.dtensor.moe.router import TokenChoiceTopKRouter as TorchRouter
from dockyard_rl.models.jax.moe import Buffer
from dockyard_rl.models.jax.moe.block import MoEBlock, per_expert_counts
from dockyard_rl.models.jax.moe.dispatch import LocalTokenDispatcher
from dockyard_rl.models.jax.moe.experts import GroupedExperts
from dockyard_rl.models.jax.moe.load_balance import (
    compute_expert_bias_delta,
    iter_lb_moe_blocks,
    update_expert_biases,
)
from dockyard_rl.models.jax.moe.router import TokenChoiceTopKRouter


def _routing_map(ids, num_experts):
    """Order-invariant one-hot (.., E) of which experts each token selects."""
    oh = np.zeros((*ids.shape[:-1], num_experts), dtype=bool)
    flat = ids.reshape(-1, ids.shape[-1])
    fm = oh.reshape(-1, num_experts)
    for t in range(fm.shape[0]):
        fm[t, flat[t]] = True
    return oh


# --- router parity (torch router is CPU-runnable) ---

@pytest.mark.parametrize("score_func", ["sigmoid", "softmax"])
@pytest.mark.parametrize("route_norm", [False, True])
def test_router_parity(score_func, route_norm):
    D, E, K = 16, 8, 2
    tr = TorchRouter(D, E, top_k=K, score_func=score_func, route_norm=route_norm, route_scale=1.3)
    jr = TokenChoiceTopKRouter(
        D, E, top_k=K, score_func=score_func, route_norm=route_norm, route_scale=1.3,
        rngs=nnx.Rngs(params=0),
    )
    jr.gate.kernel[...] = jnp.asarray(tr.gate.weight.detach().numpy().T)  # (E,D)->(D,E)

    x = np.random.default_rng(0).standard_normal((2, 5, D)).astype(np.float32)
    ts_t, ids_t, sc_t = tr(torch.from_numpy(x))
    ts_j, ids_j, sc_j = jr(jnp.asarray(x))

    # full per-expert scores are order-invariant -> direct parity
    np.testing.assert_allclose(np.asarray(sc_j), sc_t.detach().numpy(), atol=1e-5)
    # selected expert SET (routing map) matches
    np.testing.assert_array_equal(
        _routing_map(np.asarray(ids_j), E), _routing_map(ids_t.numpy(), E)
    )
    # sum of selected gating weights is order-invariant
    np.testing.assert_allclose(
        np.asarray(jnp.sum(ts_j, -1)), ts_t.sum(-1).detach().numpy(), atol=1e-5
    )


def test_router_node_limited_groups():
    D, E, K = 16, 8, 2
    tr = TorchRouter(D, E, top_k=K, score_func="sigmoid", num_expert_groups=4, num_limited_groups=2)
    jr = TokenChoiceTopKRouter(
        D, E, top_k=K, score_func="sigmoid", num_expert_groups=4, num_limited_groups=2,
        rngs=nnx.Rngs(params=0),
    )
    jr.gate.kernel[...] = jnp.asarray(tr.gate.weight.detach().numpy().T)
    x = np.random.default_rng(1).standard_normal((2, 4, D)).astype(np.float32)
    _, ids_t, _ = tr(torch.from_numpy(x))
    _, ids_j, _ = jr(jnp.asarray(x))
    np.testing.assert_array_equal(
        _routing_map(np.asarray(ids_j), E), _routing_map(ids_t.numpy(), E)
    )


def test_router_bias_shifts_choice_not_weights():
    # A large bias toward expert 0 routes every token through it, but the returned
    # gating weight for expert 0 still equals the UNBIASED score.
    D, E, K = 8, 4, 1
    jr = TokenChoiceTopKRouter(D, E, top_k=K, score_func="sigmoid", rngs=nnx.Rngs(params=0))
    x = jnp.asarray(np.random.default_rng(2).standard_normal((1, 3, D)), jnp.float32)
    bias = jnp.array([10.0, 0.0, 0.0, 0.0])
    ts, ids, sc = jr(x, bias)
    assert np.all(np.asarray(ids)[..., 0] == 0)
    np.testing.assert_allclose(np.asarray(ts[..., 0]), np.asarray(sc[..., 0]), atol=1e-6)


# --- experts grouped-GEMM vs per-expert reference ---

def _ref_experts(x_RD, counts_E, w1, w2, w3):
    """Per-expert SwiGLU over contiguous expert groups (numpy reference)."""
    out = np.zeros((x_RD.shape[0], w2.shape[1]), np.float32)
    start = 0
    for e, c in enumerate(counts_E):
        c = int(c)
        if c == 0:
            continue
        xe = x_RD[start : start + c]
        h = (xe @ w1[e].T)
        h = (h / (1.0 + np.exp(-h))) * (xe @ w3[e].T)  # silu * up
        out[start : start + c] = h @ w2[e].T
        start += c
    return out


def test_experts_grouped_gemm_matches_reference():
    D, F, E = 12, 20, 4
    disp = LocalTokenDispatcher(E, top_k=1)
    ge = GroupedExperts(D, F, E, disp, rngs=nnx.Rngs(params=0))
    rng = np.random.default_rng(3)
    counts = np.array([3, 0, 5, 2], np.int32)
    R = int(counts.sum())
    x = rng.standard_normal((R, D)).astype(np.float32)
    out_j = np.asarray(ge._experts_forward(jnp.asarray(x), jnp.asarray(counts)))
    ref = _ref_experts(
        x, counts,
        np.asarray(ge.w1_EFD[...]), np.asarray(ge.w2_EDF[...]), np.asarray(ge.w3_EFD[...]),
    )
    np.testing.assert_allclose(out_j, ref, atol=2e-4, rtol=2e-4)


# --- dispatch identity + torch parity ---

def test_dispatch_combine_identity():
    # Identity experts + score_before_experts -> combine(dispatch(x)) == x * sum_k score.
    T, D, E, K = 6, 5, 4, 2
    disp = LocalTokenDispatcher(E, top_k=K, score_before_experts=True)
    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.standard_normal((T, D)), jnp.float32)
    scores = jnp.asarray(rng.uniform(size=(T, K)), jnp.float32)
    ids = jnp.asarray(rng.integers(0, E, size=(T, K)), jnp.int32)
    routed, counts, meta = disp.dispatch(x, scores, ids)
    out = disp.combine(routed, meta, x)  # identity expert: routed_output == routed
    expected = np.asarray(x) * np.asarray(jnp.sum(scores, -1))[:, None]
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-5)
    assert int(jnp.sum(counts)) == T * K


def test_dispatch_parity_with_torch():
    T, D, E, K = 6, 5, 4, 2
    jd = LocalTokenDispatcher(E, top_k=K, score_before_experts=True)
    td = TorchDispatcher(E, top_k=K, score_before_experts=True)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((T, D)).astype(np.float32)
    scores = rng.uniform(size=(T, K)).astype(np.float32)
    ids = rng.integers(0, E, size=(T, K)).astype(np.int64)
    counts = np.bincount(ids.reshape(-1), minlength=E).astype(np.int32)

    rj, cj, mj = jd.dispatch(jnp.asarray(x), jnp.asarray(scores), jnp.asarray(ids.astype(np.int32)))
    rt, _ct, mt = td.dispatch(
        torch.from_numpy(x), torch.from_numpy(scores), torch.from_numpy(ids),
        torch.from_numpy(counts),
    )
    np.testing.assert_array_equal(np.asarray(cj), counts)
    # routed inputs are expert-sorted identically (stable sort both sides)
    np.testing.assert_allclose(np.asarray(rj), rt.numpy(), atol=1e-6)
    # identity-expert combine matches
    oj = jd.combine(rj, mj, jnp.asarray(x))
    ot = td.combine(rt, mt, torch.from_numpy(x))
    np.testing.assert_allclose(np.asarray(oj), ot.numpy(), atol=1e-6)


# --- block forward vs manual reference ---

def test_block_forward_and_counts():
    D, F, E, K = 12, 16, 4, 2
    router = TokenChoiceTopKRouter(D, E, top_k=K, score_func="sigmoid", rngs=nnx.Rngs(params=0))
    disp = LocalTokenDispatcher(E, top_k=K, score_before_experts=True)
    experts = GroupedExperts(D, F, E, disp, rngs=nnx.Rngs(params=1))
    block = MoEBlock(router, experts, num_experts=E)

    x = jnp.asarray(np.random.default_rng(6).standard_normal((2, 3, D)), jnp.float32)
    out = np.asarray(block(x))

    # manual reference: score_before_experts=True (the block's dispatcher default),
    # so the gate weight scales the INPUT -> sum_k SwiGLU_{expert_k}(score_k * x).
    ts, ids, _ = router(x)
    ts, ids = np.asarray(ts), np.asarray(ids)
    w1, w2, w3 = (np.asarray(experts.w1_EFD[...]), np.asarray(experts.w2_EDF[...]),
                  np.asarray(experts.w3_EFD[...]))
    xb = np.asarray(x)
    ref = np.zeros_like(xb)
    for b in range(xb.shape[0]):
        for t in range(xb.shape[1]):
            acc = np.zeros(D, np.float32)
            for k in range(K):
                e = int(ids[b, t, k]); xv = xb[b, t] * float(ts[b, t, k])
                h = (xv @ w1[e].T); h = (h / (1 + np.exp(-h))) * (xv @ w3[e].T)
                acc += h @ w2[e].T
            ref[b, t] = acc
    np.testing.assert_allclose(out, ref, atol=2e-4, rtol=2e-4)

    # token counter accumulated this forward equals the routing counts
    counts = np.asarray(per_expert_counts(jnp.asarray(ids), E))
    np.testing.assert_allclose(np.asarray(block.tokens_per_expert_E[...]), counts, atol=0)


# --- load balance ---

def test_bias_delta_parity_and_zero_sum():
    tokens = np.array([10.0, 2.0, 2.0, 50.0], np.float32)
    dj = np.asarray(compute_expert_bias_delta(jnp.asarray(tokens), 0.01))
    dt = torch_bias_delta(torch.from_numpy(tokens), 0.01).numpy()
    np.testing.assert_allclose(dj, dt, atol=1e-7)
    assert abs(float(dj.sum())) < 1e-6  # zero-centered


def test_update_expert_biases_steps_and_zeros():
    D, F, E, K = 8, 12, 4, 1
    router = TokenChoiceTopKRouter(D, E, top_k=K, rngs=nnx.Rngs(params=0))
    disp = LocalTokenDispatcher(E, top_k=K)
    experts = GroupedExperts(D, F, E, disp, rngs=nnx.Rngs(params=1))
    block = MoEBlock(router, experts, num_experts=E, load_balance_coeff=0.01)
    assert isinstance(block.expert_bias_E, Buffer)

    # simulate an accumulated load
    block.tokens_per_expert_E[...] = jnp.asarray([10.0, 2.0, 2.0, 50.0], jnp.float32)
    blocks = list(iter_lb_moe_blocks(block))
    assert len(blocks) == 1
    update_expert_biases(blocks)
    delta = np.asarray(compute_expert_bias_delta(jnp.array([10.0, 2.0, 2.0, 50.0]), 0.01))
    np.testing.assert_allclose(np.asarray(block.expert_bias_E[...]), delta, atol=1e-7)
    # counter zeroed for the next window
    assert float(jnp.sum(jnp.abs(block.tokens_per_expert_E[...]))) == 0.0


def test_lb_disabled_has_no_bias_buffer():
    D, F, E = 8, 12, 4
    router = TokenChoiceTopKRouter(D, E, top_k=1, rngs=nnx.Rngs(params=0))
    disp = LocalTokenDispatcher(E, top_k=1)
    experts = GroupedExperts(D, F, E, disp, rngs=nnx.Rngs(params=1))
    block = MoEBlock(router, experts, num_experts=E)
    assert block.expert_bias_E is None
    assert list(iter_lb_moe_blocks(block)) == []
