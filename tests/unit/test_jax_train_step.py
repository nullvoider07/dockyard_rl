"""J4: optax train step on a toy JAX Qwen3 — loss decrease, grad-accum, metrics.

CPU only. Checks (1) the GRPO loss decreases over a few optimizer steps on a
fixed batch, (2) microbatched grad accumulation equals the single-batch
gradient, (3) the train() metrics dict carries the expected keys.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402
nnx = pytest.importorskip("flax.nnx")
optax = pytest.importorskip("optax")

from dockyard_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
from dockyard_rl.models.jax.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from dockyard_rl.models.jax.policy_worker import JaxPolicyWorkerImpl as JaxPolicyWorker
from dockyard_rl.models.jax.train_step import accumulate_grads, global_valid_counts


def _cfg() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
    )


def _batch(seed: int = 0, B: int = 4, S: int = 9, vocab: int = 64):
    rng = np.random.default_rng(seed)
    return {
        "input_ids": jnp.asarray(rng.integers(0, vocab, size=(B, S)).astype(np.int32)),
        "advantages": jnp.asarray(rng.standard_normal((B, S)).astype(np.float32)),
        "token_mask": jnp.asarray((rng.uniform(size=(B, S)) > 0.1).astype(np.float32)),
        "sample_mask": jnp.asarray(np.ones((B,), np.float32)),
        "prev_logprobs": jnp.asarray((rng.standard_normal((B, S)) * 0.3 - 1.0).astype(np.float32)),
        "generation_logprobs": jnp.asarray((rng.standard_normal((B, S)) * 0.3 - 1.0).astype(np.float32)),
        "reference_policy_logprobs": jnp.asarray((rng.standard_normal((B, S)) * 0.3 - 1.0).astype(np.float32)),
    }


def _loss_cfg() -> ClippedPGLossConfig:
    return ClippedPGLossConfig(
        token_level_loss=True, ratio_clip_min=0.2, ratio_clip_max=0.2,
        reference_policy_kl_penalty=0.01, use_importance_sampling_correction=True,
    )


def test_loss_decreases_and_metric_keys():
    model = Qwen3ForCausalLM(_cfg(), rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    worker = JaxPolicyWorker(
        model,
        optimizer_cfg={"name": "AdamW", "kwargs": {"lr": 1e-2, "weight_decay": 0.0}},
        loss_cfg=_loss_cfg(),
        max_grad_norm=1.0,
        train_micro_batch_size=2,
    )
    batch = _batch()
    first = worker.train(batch)
    last = first
    for _ in range(6):
        last = worker.train(batch)

    assert {"loss", "grad_norm", "lr"} <= set(last.keys())
    assert np.isfinite(last["loss"]) and np.isfinite(last["grad_norm"])
    assert last["loss"] < first["loss"]  # optimizer reduces loss on its own batch


def test_grad_accumulation_equals_single_batch():
    # Microbatched accumulation must equal the full-batch gradient (global norm).
    batch = _batch(seed=3, B=4)
    cfg = _loss_cfg()

    m1 = Qwen3ForCausalLM(_cfg(), rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    gvs, gvt = global_valid_counts(batch)
    g_full, _ = accumulate_grads(m1, batch, gvs, gvt, cfg, mbs=4)   # one microbatch
    g_split, _ = accumulate_grads(m1, batch, gvs, gvt, cfg, mbs=1)  # four microbatches

    leaves_full = jax.tree_util.tree_leaves(g_full)
    leaves_split = jax.tree_util.tree_leaves(g_split)
    assert len(leaves_full) == len(leaves_split) and leaves_full
    for a, b in zip(leaves_full, leaves_split):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5, rtol=1e-5)
