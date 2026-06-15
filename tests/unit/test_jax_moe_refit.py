"""J8d: MoE refit re-expansion — fused GroupedExperts -> per-expert HF tensors.

Round-trip: load an HF Qwen3-MoE into the JAX model, then emit the refit stream
and confirm (a) the per-expert names match vLLM's checkpoint layout, (b) the
expert values are exactly the slices/unbind of the original fused HF params (so
refit inverts the loader), and (c) the router gate + a dense param map back to
their HF keys/shapes. The ep>1 gather + live vLLM load is hardware-deferred.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")
nnx = pytest.importorskip("flax.nnx")

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig as HFConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM as HFModel

from dockyard_rl.models.jax.moe.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from dockyard_rl.models.jax.moe.refit import (
    expand_grouped_expert_info,
    is_grouped_expert_key,
    iter_moe_refit_state_dict,
)
from dockyard_rl.models.jax.moe.weights import load_hf_qwen3_moe_state_dict


def _tiny():
    cfg = HFConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=4,  # pyright: ignore[reportCallIssue]
        rms_norm_eps=1e-6, rope_theta=10000.0,  # pyright: ignore[reportCallIssue]
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=8,
        decoder_sparse_step=1, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    hf = HFModel(cfg).to(torch.float32).eval()
    jcfg = Qwen3MoeConfig.from_hf_config(cfg)
    jm = Qwen3MoeForCausalLM(jcfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    load_hf_qwen3_moe_state_dict(jm, hf.state_dict(), param_dtype=jnp.float32)
    return jm, hf, cfg


def test_expand_helpers():
    assert is_grouped_expert_key("model.layers.0.mlp.experts.w1_EFD")
    assert not is_grouped_expert_key("model.layers.0.mlp.gate.weight")
    info = expand_grouped_expert_info("a.mlp.experts.w2_EDF", (3, 5, 8))
    assert info == [
        ("a.mlp.experts.0.down_proj.weight", (5, 8)),
        ("a.mlp.experts.1.down_proj.weight", (5, 8)),
        ("a.mlp.experts.2.down_proj.weight", (5, 8)),
    ]


def test_moe_refit_roundtrip():
    jm, hf, cfg = _tiny()
    hf_state = hf.state_dict()
    f = cfg.moe_intermediate_size

    emitted = {name: np.asarray(arr) for name, arr in
               iter_moe_refit_state_dict(jm, to_torch=False)}

    for li in range(cfg.num_hidden_layers):
        gu = hf_state[f"model.layers.{li}.mlp.experts.gate_up_proj"].numpy()  # (E,2F,D)
        down = hf_state[f"model.layers.{li}.mlp.experts.down_proj"].numpy()   # (E,D,F)
        for e in range(cfg.num_experts):
            g = emitted[f"model.layers.{li}.mlp.experts.{e}.gate_proj.weight"]
            u = emitted[f"model.layers.{li}.mlp.experts.{e}.up_proj.weight"]
            d = emitted[f"model.layers.{li}.mlp.experts.{e}.down_proj.weight"]
            np.testing.assert_allclose(g, gu[e, :f, :], atol=1e-6)
            np.testing.assert_allclose(u, gu[e, f:, :], atol=1e-6)
            np.testing.assert_allclose(d, down[e], atol=1e-6)
        # router gate maps back to the HF (E, D) layout
        gate = emitted[f"model.layers.{li}.mlp.gate.weight"]
        np.testing.assert_allclose(
            gate, hf_state[f"model.layers.{li}.mlp.gate.weight"].numpy(), atol=1e-6
        )

    # a dense param round-trips by HF key + shape (kernel transposed back)
    qp = emitted["model.layers.0.self_attn.q_proj.weight"]
    assert qp.shape == tuple(hf_state["model.layers.0.self_attn.q_proj.weight"].shape)
    np.testing.assert_allclose(
        qp, hf_state["model.layers.0.self_attn.q_proj.weight"].numpy(), atol=1e-6
    )
    # no fused-expert leaf names leak into the emitted stream
    assert not any(is_grouped_expert_key(k) for k in emitted)


def test_moe_refit_to_torch():
    jm, _hf, _cfg = _tiny()
    stream = dict(iter_moe_refit_state_dict(jm, to_torch=True))
    sample = stream["model.layers.0.mlp.experts.0.gate_proj.weight"]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (8, 16)  # (F, D)


def test_worker_routes_moe_refit():
    # The JAX worker detects an MoE model and routes prepare_refit_info through
    # the per-expert expansion (the J8e selection/refit wiring).
    from dockyard_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
    from dockyard_rl.models.jax.policy_worker import JaxPolicyWorkerImpl

    jm, _hf, cfg = _tiny()
    worker = JaxPolicyWorkerImpl(
        jm, optimizer_cfg={"name": "AdamW", "kwargs": {"lr": 1e-3}},
        loss_cfg=ClippedPGLossConfig(), reference_model=None, init_reference_model=False,
    )
    assert worker._is_moe is True
    info = worker.prepare_refit_info()
    assert info is not None
    # per-expert HF names present; no fused leaf names
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in info
    assert "model.layers.0.mlp.gate.weight" in info
    assert not any(is_grouped_expert_key(k) for k in info)
    shape, _dtype = info["model.layers.0.mlp.experts.0.gate_proj.weight"]
    assert tuple(shape) == (cfg.moe_intermediate_size, cfg.hidden_size)
