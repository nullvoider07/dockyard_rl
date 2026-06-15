"""J5: refit name-map + value round-trip on CPU (NCCL broadcast is HV).

HF model -> J1 loader -> NNX -> J5 emit must reproduce the original HF
state-dict names, shapes, and values exactly (the refit name/transpose set is
the inverse of the loader). Also checks prepare_refit_info's HF-layout shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")
nnx = pytest.importorskip("flax.nnx")

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config as HFQwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as HFQwen3ForCausalLM

from dockyard_rl.models.jax.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from dockyard_rl.models.jax.refit import iter_refit_state_dict, prepare_refit_info
from dockyard_rl.models.jax.weights import load_hf_state_dict


def _hf_model():
    cfg = HFQwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, tie_word_embeddings=False,
        rope_theta=10000.0,  # pyright: ignore[reportCallIssue]
    )
    torch.manual_seed(0)
    return HFQwen3ForCausalLM(cfg).to(torch.float32).eval(), cfg


def _jax_loaded(hf_model, hf_cfg):
    jcfg = Qwen3Config.from_hf_config(hf_cfg)
    jm = Qwen3ForCausalLM(jcfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    load_hf_state_dict(jm, hf_model.state_dict(), param_dtype=jnp.float32)
    return jm


def test_refit_reproduces_hf_state_dict():
    hf_model, hf_cfg = _hf_model()
    hf_sd = {k: v.detach().numpy() for k, v in hf_model.state_dict().items()}
    jm = _jax_loaded(hf_model, hf_cfg)

    emitted = dict(iter_refit_state_dict(jm))
    # Same names, same shapes, same values as the original HF state dict.
    assert set(emitted.keys()) == set(hf_sd.keys())
    for name, t in emitted.items():
        arr = t.detach().numpy()
        assert arr.shape == hf_sd[name].shape, name
        np.testing.assert_allclose(arr, hf_sd[name], atol=1e-6, rtol=1e-6, err_msg=name)


def test_prepare_refit_info_matches_hf_shapes():
    hf_model, hf_cfg = _hf_model()
    jm = _jax_loaded(hf_model, hf_cfg)
    info = prepare_refit_info(jm)
    hf_sd = hf_model.state_dict()

    assert set(info.keys()) == set(hf_sd.keys())
    for name, (shape, dtype_name) in info.items():
        assert tuple(shape) == tuple(hf_sd[name].shape), name
        assert dtype_name == "float32"


def test_jax_to_torch_value_roundtrip():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 3)).astype(np.float32)
    from dockyard_rl.models.jax.refit import _jax_to_torch

    t = _jax_to_torch(jnp.asarray(a))
    np.testing.assert_array_equal(t.detach().numpy(), a)
    # transposed (kernel) orientation also round-trips
    t2 = _jax_to_torch(jnp.asarray(a).T)
    np.testing.assert_array_equal(t2.detach().numpy(), a.T)
