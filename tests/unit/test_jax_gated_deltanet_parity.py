"""J11b: Qwen3NextGatedDeltaNet NNX module — CPU parity vs the HF torch module.

Builds the HF `Qwen3NextGatedDeltaNet` (pure-torch fallback path on CPU), copies
its weights into the JAX module, and asserts the full token-mixer forward matches
on a tiny config exercising GQA (num_value_heads > num_key_heads). Tolerance tracks
J11a (the JAX scan vs HF's chunked training kernel agree to ~2e-4).
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")
nnx = pytest.importorskip("flax.nnx")

from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet as TorchGDN

from dockyard_rl.models.jax.linear_attn.gated_deltanet import Qwen3NextGatedDeltaNet


def _cfg() -> Qwen3NextConfig:
    return Qwen3NextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6,
        hidden_act="silu", linear_conv_kernel_dim=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
    )


def _copy_weights(jmod: Qwen3NextGatedDeltaNet, tmod: TorchGDN) -> None:
    def t(p):
        return jnp.asarray(p.detach().numpy())

    jmod.in_proj_qkvz.kernel[...] = t(tmod.in_proj_qkvz.weight).T
    jmod.in_proj_ba.kernel[...] = t(tmod.in_proj_ba.weight).T
    jmod.out_proj.kernel[...] = t(tmod.out_proj.weight).T
    jmod.conv_weight[...] = t(tmod.conv1d.weight).squeeze(1)  # (C,1,K) -> (C,K)
    jmod.dt_bias[...] = t(tmod.dt_bias)
    jmod.A_log[...] = t(tmod.A_log)
    jmod.norm.weight[...] = t(tmod.norm.weight)


def test_gated_deltanet_module_parity():
    cfg = _cfg()
    tmod = TorchGDN(cfg, layer_idx=0).eval()
    jmod = Qwen3NextGatedDeltaNet(cfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    _copy_weights(jmod, tmod)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 12, cfg.hidden_size)).astype(np.float32)
    with torch.no_grad():
        ref = tmod(torch.from_numpy(x), cache_params=None, attention_mask=None).numpy()
    got = np.asarray(jmod(jnp.asarray(x)))

    np.testing.assert_allclose(got, ref, atol=2e-4, rtol=2e-4)
