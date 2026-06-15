"""J9: JaxPolicyWorker implements the PolicyInterface/Colocatable surface (CPU).

Exercises the methods the GRPO driver calls (train with a real ClippedPGLossFn,
get_logprobs/reference/topk, save_checkpoint, prepare_refit_info, lifecycle
no-ops, shutdown) and confirms the deferred/HV methods raise clearly. The live
Ray-actor construction + jax.distributed + NCCL refit are GPU/cluster (HV).
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")

from dockyard_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn
from dockyard_rl.models.jax.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from dockyard_rl.models.jax.policy_worker import JaxPolicyWorkerImpl as JaxPolicyWorker


def _cfg():
    return Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
    )


def _model(seed=0):
    return Qwen3ForCausalLM(_cfg(), rngs=nnx.Rngs(params=seed), param_dtype=jnp.float32)


def _loss_cfg():
    return ClippedPGLossConfig(
        token_level_loss=True, reference_policy_kl_penalty=0.01,
        use_importance_sampling_correction=True,
    )


def _batch(B=4, S=9, vocab=64):
    rng = np.random.default_rng(0)
    return {
        "input_ids": jnp.asarray(rng.integers(0, vocab, size=(B, S)).astype(np.int32)),
        "advantages": jnp.asarray(rng.standard_normal((B, S)).astype(np.float32)),
        "token_mask": jnp.asarray(np.ones((B, S), np.float32)),
        "sample_mask": jnp.asarray(np.ones((B,), np.float32)),
        "prev_logprobs": jnp.asarray((rng.standard_normal((B, S)) * 0.3).astype(np.float32)),
        "generation_logprobs": jnp.asarray((rng.standard_normal((B, S)) * 0.3).astype(np.float32)),
        "reference_policy_logprobs": jnp.asarray((rng.standard_normal((B, S)) * 0.3).astype(np.float32)),
    }


def test_loss_cfg_derived_from_loss_fn_matches_stored():
    # train(loss_fn=ClippedPGLossFn) must equal train(loss_cfg=same config).
    cfg = _loss_cfg()
    batch = _batch()
    w_stored = JaxPolicyWorker(_model(0), optimizer_cfg={"name": "AdamW", "kwargs": {"lr": 0.0}},
                               loss_cfg=cfg, train_micro_batch_size=2)
    w_fn = JaxPolicyWorker(_model(0), optimizer_cfg={"name": "AdamW", "kwargs": {"lr": 0.0}},
                           loss_cfg=None, train_micro_batch_size=2)
    m_stored = w_stored.train(batch, eval_mode=True)
    m_fn = w_fn.train(batch, loss_fn=ClippedPGLossFn(cfg), eval_mode=True)
    assert abs(m_stored["loss"] - m_fn["loss"]) < 1e-6


def test_interface_surface(tmp_path):
    w = JaxPolicyWorker(_model(0), optimizer_cfg={"name": "AdamW", "kwargs": {"lr": 1e-3}},
                        loss_cfg=_loss_cfg(), reference_model=_model(0),
                        max_grad_norm=1.0, train_micro_batch_size=2, logprob_batch_size=2)
    batch = _batch()

    w.prepare_for_training()
    metrics = w.train(batch, loss_fn=ClippedPGLossFn(_loss_cfg()))
    assert {"loss", "grad_norm", "lr"} <= set(metrics)
    w.finish_training()

    assert w.get_logprobs(batch)["logprobs"].shape == (4, 9)
    assert w.get_reference_policy_logprobs(batch)["reference_logprobs"].shape == (4, 9)
    assert w.get_topk_logits(batch, k=3)["topk_logits"].shape == (4, 8, 3)

    info = w.prepare_refit_info()
    assert info is not None
    assert "lm_head.weight" in info and "model.embed_tokens.weight" in info

    w.save_checkpoint(str(tmp_path / "ck"))

    # lifecycle no-ops + shutdown
    assert w.prepare_for_generation() is True
    assert w.prepare_for_lp_inference() is None
    assert w.offload_before_refit() is None
    assert w.offload_after_refit() is None
    assert w.shutdown() is True


def test_deferred_methods_raise():
    w = JaxPolicyWorker(_model(0), optimizer_cfg={"name": "AdamW", "kwargs": {"lr": 1e-3}},
                        loss_cfg=_loss_cfg())
    for call in (
        lambda: w.calibrate_qkv_fp8_scales(_batch()),
        lambda: w.train_from_meta(None),
        lambda: w.get_logprobs_from_meta(None),
        lambda: w.init_collective("127.0.0.1", 1234, 2, train_world_size=1),
        lambda: w.broadcast_weights_for_collective(),  # no init_collective -> RuntimeError
    ):
        with pytest.raises((NotImplementedError, RuntimeError)):
            call()
