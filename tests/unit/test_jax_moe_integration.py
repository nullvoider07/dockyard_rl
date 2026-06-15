"""J8 integration: aux-loss-free load-balance step wired into the train step, and
the EP-dispatcher wiring helper.

CPU only. Validates (1) a real ``worker.train`` step accumulates per-expert load
through ``value_and_grad`` (buffer mutation propagates back), steps ``expert_bias_E``
zero-centeredly, and zeroes the counter for the next window; (2) eval_mode does not
step the bias; (3) ``_wire_ep_dispatchers`` installs the ep axis on all-to-all
dispatchers only. The live EP collective is hardware-deferred (HV-29).
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
optax = pytest.importorskip("optax")

from dockyard_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
from dockyard_rl.models.jax.moe.load_balance import iter_lb_moe_blocks, update_expert_biases
from dockyard_rl.models.jax.moe.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from dockyard_rl.models.jax.policy_worker import JaxPolicyWorkerImpl as JaxPolicyWorker
from dockyard_rl.models.jax.policy_worker import _wire_ep_dispatchers


def _moe_cfg() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4, rms_norm_eps=1e-6,
        rope_theta=10000.0, tie_word_embeddings=False, num_experts=6, num_experts_per_tok=2,
        moe_intermediate_size=8, decoder_sparse_step=1,
    )


def _batch(seed: int = 0, B: int = 4, S: int = 8, vocab: int = 32):
    rng = np.random.default_rng(seed)
    return {
        "input_ids": jnp.asarray(rng.integers(0, vocab, size=(B, S)).astype(np.int32)),
        "advantages": jnp.asarray(rng.standard_normal((B, S)).astype(np.float32)),
        "token_mask": jnp.asarray((rng.uniform(size=(B, S)) > 0.1).astype(np.float32)),
        "sample_mask": jnp.asarray(np.ones((B,), np.float32)),
        "prev_logprobs": jnp.asarray((rng.standard_normal((B, S)) * 0.3 - 1.0).astype(np.float32)),
        "generation_logprobs": jnp.asarray((rng.standard_normal((B, S)) * 0.3 - 1.0).astype(np.float32)),
        "reference_policy_logprobs": jnp.asarray(
            (rng.standard_normal((B, S)) * 0.3 - 1.0).astype(np.float32)
        ),
    }


def _loss_cfg() -> ClippedPGLossConfig:
    return ClippedPGLossConfig(
        token_level_loss=True, ratio_clip_min=0.2, ratio_clip_max=0.2,
        reference_policy_kl_penalty=0.01, use_importance_sampling_correction=True,
    )


def _worker(coeff: float) -> JaxPolicyWorker:
    model = Qwen3MoeForCausalLM(
        _moe_cfg(), rngs=nnx.Rngs(params=0), param_dtype=jnp.float32,
        load_balance_coeff=coeff, token_dispatcher="local",
    )
    return JaxPolicyWorker(
        model,
        optimizer_cfg={"name": "AdamW", "kwargs": {"lr": 1e-2, "weight_decay": 0.0}},
        loss_cfg=_loss_cfg(),
        max_grad_norm=1.0,
        train_micro_batch_size=2,
    )


def test_train_step_updates_expert_bias_and_zeroes_counter():
    coeff = 0.01
    worker = _worker(coeff)
    assert worker._lb_update_fn is not None  # LB updater discovered for MoE
    blocks = list(iter_lb_moe_blocks(worker.model))
    assert blocks, "load-balanced MoEBlocks expected"
    for b in blocks:
        assert b.expert_bias_E is not None

    before = [np.asarray(b.expert_bias_E[...]) for b in blocks if b.expert_bias_E is not None]
    metrics = worker.train(_batch())

    after = [np.asarray(b.expert_bias_E[...]) for b in blocks if b.expert_bias_E is not None]
    for b0, b1, block in zip(before, after, blocks):
        # bias moved (counter accumulated through value_and_grad => non-zero load)
        assert not np.allclose(b0, b1), "expert_bias_E did not update"
        # sign step (coeff * sign(mean - load)) then zero-centered: distinct values
        # are the sign classes shifted by the same mean, so their spacing is the
        # coeff grid, and the whole delta sums to ~0.
        delta = b1 - b0
        grid = np.unique(np.round(delta / coeff, 4))
        spacing = grid - grid.min()
        np.testing.assert_allclose(spacing, np.round(spacing), atol=1e-3)
        assert abs(float(delta.sum())) < 1e-5, "bias delta not zero-centered"
        # counter reset for the next window
        np.testing.assert_allclose(np.asarray(block.tokens_per_expert_E[...]), 0.0, atol=0)
        assert block.expert_bias_E is not None
    assert np.isfinite(metrics["loss"])


def test_eval_mode_does_not_step_expert_bias():
    worker = _worker(0.02)
    blocks = list(iter_lb_moe_blocks(worker.model))
    for b in blocks:
        assert b.expert_bias_E is not None
    before = [np.asarray(b.expert_bias_E[...]).copy() for b in blocks if b.expert_bias_E is not None]
    worker.train(_batch(), eval_mode=True)
    for b0, block in zip(before, blocks):
        assert block.expert_bias_E is not None
        np.testing.assert_array_equal(np.asarray(block.expert_bias_E[...]), b0)


def test_dense_worker_has_no_lb_updater():
    from dockyard_rl.models.jax.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(cfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    worker = JaxPolicyWorker(
        model, optimizer_cfg={"name": "AdamW", "kwargs": {"lr": 1e-2}}, loss_cfg=_loss_cfg(),
    )
    assert worker._is_moe is False
    assert worker._lb_update_fn is None


def test_load_balance_survives_nnx_jit():
    # HV-31b de-risk: under nnx.jit the forward propagates the tokens_per_expert_E
    # Buffer mutation AND update_expert_biases runs inside jit (nnx threads the
    # buffer state), so J10's jitted train step needs no functionalization of the
    # load-balance updater. (The cross-replica psum reduce stays GPU-only, HV-31a.)
    coeff = 0.01
    model = Qwen3MoeForCausalLM(
        _moe_cfg(), rngs=nnx.Rngs(params=0), param_dtype=jnp.float32, load_balance_coeff=coeff,
    )
    ids = jnp.asarray(np.random.default_rng(0).integers(0, 32, size=(2, 8)), jnp.int32)
    block = list(iter_lb_moe_blocks(model))[0]
    assert block.expert_bias_E is not None

    @nnx.jit
    def fwd(m, x):
        return m(x)

    fwd(model, ids)  # jitted forward
    assert float(jnp.sum(block.tokens_per_expert_E[...])) > 0.0  # counter propagated through jit

    bias_before = np.asarray(block.expert_bias_E[...]).copy()

    @nnx.jit
    def lb_step(m):
        update_expert_biases(list(iter_lb_moe_blocks(m)), reduce_tokens_fn=None)

    lb_step(model)  # the updater runs *inside* jit
    assert not np.allclose(np.asarray(block.expert_bias_E[...]), bias_before)  # bias stepped
    np.testing.assert_allclose(np.asarray(block.tokens_per_expert_E[...]), 0.0, atol=0)  # zeroed


def test_wire_ep_dispatchers_only_wires_alltoall():
    # "local" dispatcher model: nothing to wire.
    local_model = Qwen3MoeForCausalLM(
        _moe_cfg(), rngs=nnx.Rngs(params=0), token_dispatcher="local",
    )
    assert _wire_ep_dispatchers(local_model, "ep", 2) == 0

    # "alltoall" dispatcher model: every sparse layer's dispatcher gets wired.
    a2a_model = Qwen3MoeForCausalLM(
        _moe_cfg(), rngs=nnx.Rngs(params=0), token_dispatcher="alltoall",
    )
    wired = _wire_ep_dispatchers(a2a_model, "ep", 2)
    assert wired == 2  # decoder_sparse_step=1, num_hidden_layers=2 -> 2 sparse layers
    from dockyard_rl.models.jax.moe.alltoall import AllToAllTokenDispatcher
    from dockyard_rl.models.jax.moe.experts import GroupedExperts

    for _p, m in nnx.iter_modules(a2a_model):
        if isinstance(m, GroupedExperts):
            disp = m.dispatcher
            assert isinstance(disp, AllToAllTokenDispatcher)
            assert disp.ep_axis == "ep" and disp.ep_size == 2
            assert disp._ep_enabled() is True
