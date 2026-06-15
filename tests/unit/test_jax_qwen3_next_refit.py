"""J11 refit: Qwen3-Next NNX params -> HF-named tensors — CPU round-trip.

The refit name-map must be the exact inverse of the loader: load an HF state dict,
emit the refit stream, and it reproduces every HF tensor (non-expert keys directly;
fused experts via the per-expert expansion the vLLM FusedMoE re-fuses). Also checks
`prepare_qwen3_next_refit_info` declares the same names/shapes the stream yields, and
that the worker routes qwen3_next refit. Live vLLM broadcast + ep>1 gather are HV.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")
nnx = pytest.importorskip("flax.nnx")

from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig as HFConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM as HFModel

from dockyard_rl.models.jax.linear_attn.refit import (
    iter_qwen3_next_refit_state_dict,
    prepare_qwen3_next_refit_info,
)
from dockyard_rl.models.jax.linear_attn.weights import load_hf_qwen3_next_state_dict
from dockyard_rl.models.jax.models.qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM


def _hf_cfg(mlp_only_layers):
    cfg = HFConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
        hidden_act="silu", linear_conv_kernel_dim=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, decoder_sparse_step=1, norm_topk_prob=True,
        tie_word_embeddings=False, mlp_only_layers=mlp_only_layers,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _is_expanded_expert(name: str) -> bool:
    parts = name.split(".")
    return "experts" in parts and parts[parts.index("experts") + 1].isdigit()


@pytest.mark.parametrize("mlp_only_layers", [[], [2]])
def test_qwen3_next_refit_roundtrip(mlp_only_layers):
    hf_cfg = _hf_cfg(mlp_only_layers)
    hf = HFModel(hf_cfg).eval()
    hf_sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}

    jcfg = Qwen3NextConfig.from_hf_config(hf_cfg)
    jmodel = Qwen3NextForCausalLM(jcfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    load_hf_qwen3_next_state_dict(jmodel, hf.state_dict(), param_dtype=jnp.float32)

    refit = {name: np.asarray(arr) for name, arr in iter_qwen3_next_refit_state_dict(jmodel, to_torch=False)}
    f = hf_cfg.moe_intermediate_size

    # 1. non-expert keys reproduce HF exactly (name, shape, value)
    non_expert = {n: a for n, a in refit.items() if not _is_expanded_expert(n)}
    hf_non_expert = {k for k in hf_sd if not (k.endswith(".experts.gate_up_proj") or k.endswith(".experts.down_proj"))}
    assert set(non_expert) == hf_non_expert, "name-map mismatch (non-expert keys)"
    for name, arr in non_expert.items():
        np.testing.assert_allclose(arr, hf_sd[name], atol=0, err_msg=f"value drift at {name}")

    # 2. expanded experts re-fuse to the HF fused tensors (gate=:F, up=F:, down passthrough)
    n_experts = hf_cfg.num_experts
    for key, fused in hf_sd.items():
        if key.endswith(".experts.gate_up_proj"):
            base = key[: -len(".gate_up_proj")]
            for e in range(n_experts):
                np.testing.assert_allclose(refit[f"{base}.{e}.gate_proj.weight"], fused[e, :f, :], atol=0)
                np.testing.assert_allclose(refit[f"{base}.{e}.up_proj.weight"], fused[e, f:, :], atol=0)
        elif key.endswith(".experts.down_proj"):
            base = key[: -len(".down_proj")]
            for e in range(n_experts):
                np.testing.assert_allclose(refit[f"{base}.{e}.down_proj.weight"], fused[e], atol=0)

    # 3. linear-attn specials: conv1d weight is re-expanded to (C,1,K); scalars keep their names
    conv_keys = [n for n in refit if n.endswith(".conv1d.weight")]
    assert conv_keys and all(refit[n].ndim == 3 and refit[n].shape[1] == 1 for n in conv_keys)
    assert any(n.endswith(".A_log") for n in refit) and any(n.endswith(".dt_bias") for n in refit)

    # 4. prepare_*_refit_info declares the same names + shapes the stream yields
    info = prepare_qwen3_next_refit_info(jmodel)
    assert set(info.keys()) == set(refit.keys())
    for name, (shape, _dt) in info.items():
        assert tuple(shape) == refit[name].shape, f"declared shape != streamed at {name}"


def test_worker_routes_qwen3_next_refit():
    from dockyard_rl.models.jax.policy_worker import JaxPolicyWorkerImpl

    hf_cfg = _hf_cfg([])
    hf = HFModel(hf_cfg).eval()
    jcfg = Qwen3NextConfig.from_hf_config(hf_cfg)
    jmodel = Qwen3NextForCausalLM(jcfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    load_hf_qwen3_next_state_dict(jmodel, hf.state_dict())
    worker = JaxPolicyWorkerImpl(jmodel, init_optimizer=False)

    info = worker.prepare_refit_info()  # no longer raises; routes to the qwen3_next map
    assert info is not None and any(".conv1d.weight" in k for k in info)
    assert any(".experts.0.gate_proj.weight" in k for k in info)
