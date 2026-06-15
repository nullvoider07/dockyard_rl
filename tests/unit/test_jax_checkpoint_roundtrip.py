"""J7: Orbax checkpoint roundtrip + HF-safetensors export re-loadability.

CPU only. Save model+optimizer+step, restore into a fresh model/optimizer, and
assert param + step + optimizer-moment equality. Separately, export HF
safetensors and re-load via the J1 loader to confirm cross-backend weight
interchange.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")
pytest.importorskip("orbax.checkpoint")

from dockyard_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
from dockyard_rl.models.jax.checkpoint import export_hf_safetensors, restore_checkpoint
from dockyard_rl.models.jax.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from dockyard_rl.models.jax.policy_worker import JaxPolicyWorkerImpl as JaxPolicyWorker
from dockyard_rl.models.jax.weights import load_hf_state_dict


def _cfg() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
    )


def _worker(seed: int) -> JaxPolicyWorker:
    model = Qwen3ForCausalLM(_cfg(), rngs=nnx.Rngs(params=seed), param_dtype=jnp.float32)
    return JaxPolicyWorker(
        model, optimizer_cfg={"name": "AdamW", "kwargs": {"lr": 1e-2}},
        loss_cfg=ClippedPGLossConfig(reference_policy_kl_penalty=0.0),
        max_grad_norm=1.0, train_micro_batch_size=2,
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
    }


def _params(model):
    return [np.asarray(v[...]) for _, v in nnx.to_flat_state(nnx.state(model))]


def test_checkpoint_roundtrip(tmp_path):
    w = _worker(0)
    batch = _batch()
    for _ in range(3):
        w.train(batch)  # advance step + populate optimizer moments
    saved_params = _params(w.model)
    saved_step = w._step

    ckpt_dir = str(tmp_path / "ckpt0")
    w.save_checkpoint(ckpt_dir)

    # Fresh worker with different init; restore into it.
    w2 = _worker(99)
    pre = _params(w2.model)
    assert not np.allclose(pre[0], saved_params[0])  # genuinely different before restore
    step = restore_checkpoint(ckpt_dir, w2.model, optimizer=w2.optimizer)

    assert step == saved_step
    for a, b in zip(saved_params, _params(w2.model)):
        np.testing.assert_allclose(a, b, atol=0, rtol=0)

    # Optimizer moments restored: one more identical step on both gives identical params.
    w2._step = step
    w.train(batch)
    w2.train(batch)
    for a, b in zip(_params(w.model), _params(w2.model)):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)


def test_hf_export_reloadable(tmp_path):
    w = _worker(0)
    st_path = str(tmp_path / "model.safetensors")
    export_hf_safetensors(w.model, st_path)

    from safetensors.torch import load_file

    sd = load_file(st_path)
    fresh = Qwen3ForCausalLM(_cfg(), rngs=nnx.Rngs(params=123), param_dtype=jnp.float32)
    load_hf_state_dict(fresh, sd, param_dtype=jnp.float32)

    for a, b in zip(_params(w.model), _params(fresh)):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
