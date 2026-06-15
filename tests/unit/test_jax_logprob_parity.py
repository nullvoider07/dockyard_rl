"""J6: logprob ingestion seam — spec, position-0=0 convention, torch parity.

The JAX worker's get_logprobs must match a direct torch reference (log_softmax +
gather + prepend-zero, as dtensor_policy_worker.py:1257 assembles it), honor the
[B,S] shape with logprobs[:,0]==0, length-mask padding, and produce the
reference/top-k spec shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")
nnx = pytest.importorskip("flax.nnx")

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config as HFQwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as HFQwen3ForCausalLM

from dockyard_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
from dockyard_rl.models.jax.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from dockyard_rl.models.jax.policy_worker import JaxPolicyWorkerImpl as JaxPolicyWorker
from dockyard_rl.models.jax.weights import load_hf_state_dict


def _models():
    cfg = HFQwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, tie_word_embeddings=False,
        rope_theta=10000.0,  # pyright: ignore[reportCallIssue]
    )
    torch.manual_seed(0)
    hf = HFQwen3ForCausalLM(cfg).to(torch.float32).eval()
    jm = Qwen3ForCausalLM(Qwen3Config.from_hf_config(cfg), rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    load_hf_state_dict(jm, hf.state_dict(), param_dtype=jnp.float32)
    return hf, jm, cfg


def _worker(jm, temperature: float = 1.0):
    return JaxPolicyWorker(
        jm, optimizer_cfg={"name": "AdamW", "kwargs": {"lr": 1e-3}},
        loss_cfg=ClippedPGLossConfig(), reference_model=jm, logprob_batch_size=2,
        generation_temperature=temperature,
    )


def _torch_ref_logprobs(hf, ids, temperature: float = 1.0):
    with torch.no_grad():
        logits = hf(input_ids=torch.from_numpy(ids)).logits.float()
    if temperature != 1.0:
        logits = logits / temperature
    lp = torch.log_softmax(logits[:, :-1], dim=-1)
    lp = lp.gather(-1, torch.from_numpy(ids)[:, 1:].unsqueeze(-1)).squeeze(-1)
    return torch.cat([torch.zeros_like(lp[:, :1]), lp], dim=1).numpy()  # [B, S]


def test_get_logprobs_parity_and_convention():
    hf, jm, cfg = _models()
    worker = _worker(jm)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.vocab_size, size=(4, 9)).astype(np.int64)

    out = worker.get_logprobs({"input_ids": ids})["logprobs"].numpy()
    ref = _torch_ref_logprobs(hf, ids)

    assert out.shape == (4, 9)
    assert np.allclose(out[:, 0], 0.0)  # position-0 convention
    np.testing.assert_allclose(out, ref, atol=2e-4, rtol=2e-4)


def test_get_logprobs_temperature_scaling():
    # Generation temperature != 1.0 must divide the logits before log-softmax,
    # matching the torch worker's _apply_temperature_scaling.
    hf, jm, cfg = _models()
    worker = _worker(jm, temperature=0.7)
    ids = np.random.default_rng(4).integers(0, cfg.vocab_size, size=(2, 9)).astype(np.int64)
    out = worker.get_logprobs({"input_ids": ids})["logprobs"].numpy()
    ref = _torch_ref_logprobs(hf, ids, temperature=0.7)
    np.testing.assert_allclose(out, ref, atol=2e-4, rtol=2e-4)
    # and it actually differs from the unscaled logprobs
    assert not np.allclose(out, _torch_ref_logprobs(hf, ids, temperature=1.0), atol=1e-3)


def test_topk_or_topp_filtering_rejected():
    # The JAX worker has no filtered-logprob path; requesting top-k/top-p filtering
    # of training logits must raise, not silently diverge from the torch filtered
    # logprobs. Detected at generation-config parse on the production __init__ path.
    from dockyard_rl.models.jax.policy_worker import _temperature_from_generation
    with pytest.raises(NotImplementedError):
        _temperature_from_generation({"temperature": 1.0, "top_p": 0.9, "top_k": None})
    with pytest.raises(NotImplementedError):
        _temperature_from_generation({"temperature": 1.0, "top_p": 1.0, "top_k": 50})
    # plain temperature-only generation config parses fine
    assert _temperature_from_generation({"temperature": 0.8, "top_p": 1.0, "top_k": None}) == 0.8


def test_reference_logprobs_spec():
    hf, jm, cfg = _models()
    worker = _worker(jm)
    ids = np.random.default_rng(1).integers(0, cfg.vocab_size, size=(2, 7)).astype(np.int64)
    out = worker.get_reference_policy_logprobs({"input_ids": ids})
    assert "reference_logprobs" in out
    assert out["reference_logprobs"].shape == (2, 7)


def test_length_mask_zeros_padding():
    hf, jm, cfg = _models()
    worker = _worker(jm)
    ids = np.random.default_rng(2).integers(0, cfg.vocab_size, size=(2, 8)).astype(np.int64)
    lengths = np.array([8, 5], dtype=np.int64)
    out = worker.get_logprobs({"input_ids": ids, "input_lengths": lengths})["logprobs"].numpy()
    assert np.allclose(out[1, 5:], 0.0)        # padded tail zeroed
    assert not np.allclose(out[1, 1:5], 0.0)   # valid region non-zero


def test_topk_logits_spec():
    hf, jm, cfg = _models()
    worker = _worker(jm)
    ids = np.random.default_rng(3).integers(0, cfg.vocab_size, size=(2, 6)).astype(np.int64)
    out = worker.get_topk_logits({"input_ids": ids}, k=5)
    assert out["topk_logits"].shape == (2, 5, 5)
    assert out["topk_indices"].shape == (2, 5, 5)
    # indices point to the argmax-ish top logits; values are sorted descending
    vals = out["topk_logits"].numpy()
    assert np.all(np.diff(vals, axis=-1) <= 1e-5)
