"""J8c: EP all-to-all dispatch — pure index math + EP-disabled local fallback.

The ``ragged_all_to_all`` collective needs a real multi-device ``ep`` mesh, so its
live numerics are hardware-deferred. CPU-validatable here: the rank-major <->
expert-major permutation index math (against a hand-computed reference and as an
inverse round-trip) and that, with no EP wired, the dispatcher is byte-identical
to the local EP=1 path.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
jax = pytest.importorskip("jax")

from dockyard_rl.models.jax.moe.alltoall import (
    AllToAllTokenDispatcher,
    expert_major_permutation,
    local_expert_counts,
)
from dockyard_rl.models.jax.moe.dispatch import LocalTokenDispatcher


def test_expert_major_permutation_reference():
    # ep=2, le=2. rank-major received counts [r0e0, r0e1, r1e0, r1e1] = [2,1,1,3].
    # expert-major wants e0(r0,r1) then e1(r0,r1): input idx [0,1, 3, 2, 4,5,6].
    counts = jnp.asarray([2, 1, 1, 3], jnp.int32)
    perm = np.asarray(expert_major_permutation(counts, ep_size=2))
    np.testing.assert_array_equal(perm, [0, 1, 3, 2, 4, 5, 6])
    # per-local-expert totals: e0=2+1=3, e1=1+3=4
    np.testing.assert_array_equal(np.asarray(local_expert_counts(counts, 2)), [3, 4])


def test_permute_is_a_permutation_and_invertible():
    rng = np.random.default_rng(0)
    counts = jnp.asarray(rng.integers(0, 5, size=8), jnp.int32)  # ep=4, le=2
    total = int(counts.sum())
    perm = expert_major_permutation(counts, ep_size=4)
    # a valid permutation of range(total)
    np.testing.assert_array_equal(np.sort(np.asarray(perm)), np.arange(total))
    # round-trip: gather then scatter-back recovers the original rows
    x = jnp.asarray(rng.standard_normal((total, 3)), jnp.float32)
    gathered = x[perm]
    restored = jnp.zeros_like(x).at[perm].set(gathered)
    np.testing.assert_allclose(np.asarray(restored), np.asarray(x), atol=0)


def test_ep_disabled_matches_local_dispatcher():
    T, D, E, K = 6, 5, 4, 2
    a2a = AllToAllTokenDispatcher(E, top_k=K, score_before_experts=True)  # not wired -> EP off
    local = LocalTokenDispatcher(E, top_k=K, score_before_experts=True)
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((T, D)), jnp.float32)
    scores = jnp.asarray(rng.uniform(size=(T, K)), jnp.float32)
    ids = jnp.asarray(rng.integers(0, E, size=(T, K)), jnp.int32)

    ra, ca, ma = a2a.dispatch(x, scores, ids)
    rl, cl, ml = local.dispatch(x, scores, ids)
    np.testing.assert_allclose(np.asarray(ra), np.asarray(rl), atol=0)
    np.testing.assert_array_equal(np.asarray(ca), np.asarray(cl))
    # identity-expert combine equals x * sum_k score (local fallback semantics)
    out = a2a.combine(ra, ma, x)
    expected = np.asarray(x) * np.asarray(jnp.sum(scores, -1))[:, None]
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-5)


def test_wire_ep_validation():
    a2a = AllToAllTokenDispatcher(8, top_k=2)
    a2a.wire_ep("ep", 4)  # 8 % 4 == 0 ok
    assert a2a._ep_enabled() is True
    a2a.wire_ep(None, 1)  # disable
    assert a2a._ep_enabled() is False
    with pytest.raises(ValueError):
        a2a.wire_ep("ep", 3)  # 8 % 3 != 0


def test_model_alltoall_dispatcher_unwired_matches_local():
    # Selecting the alltoall dispatcher but leaving ep unwired (single device) must
    # be byte-identical to the local dispatcher — the plumbing + fallback are correct.
    nnx = pytest.importorskip("flax.nnx")
    from dockyard_rl.models.jax.moe.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4, rms_norm_eps=1e-6,
        rope_theta=10000.0, tie_word_embeddings=False, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=8, decoder_sparse_step=1,
    )
    ml = Qwen3MoeForCausalLM(cfg, rngs=nnx.Rngs(params=0), token_dispatcher="local")
    ma = Qwen3MoeForCausalLM(cfg, rngs=nnx.Rngs(params=0), token_dispatcher="alltoall")
    ids = jnp.asarray(np.random.default_rng(0).integers(0, 32, size=(2, 6)), jnp.int32)
    np.testing.assert_allclose(np.asarray(ml(ids)), np.asarray(ma(ids)), atol=1e-6)
