"""J11c: full Qwen3-Next hybrid model — CPU forward-logits parity vs HF.

Builds a tiny HF `Qwen3NextForCausalLM` (eager attention, pure-torch linear-attn
fallback), loads its weights into the JAX model, and matches next-token logits.
The config mixes linear- and full-attention layers (default interval) and a dense
MLP layer (`mlp_only_layers`) alongside sparse MoE layers, so a single forward
exercises every block type + the gated shared expert + partial RoPE + (1+weight)
RMSNorm. The whole MoE compute core reuses J8 (HF-logit-parity-proven there).
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")
nnx = pytest.importorskip("flax.nnx")

from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig as HFConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM as HFModel

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


@pytest.mark.parametrize("mlp_only_layers", [[], [2]])
def test_qwen3_next_forward_logits_parity(mlp_only_layers):
    hf_cfg = _hf_cfg(mlp_only_layers=mlp_only_layers)
    lt = hf_cfg.layer_types or []
    assert "linear_attention" in lt and "full_attention" in lt  # both block types exercised
    hf = HFModel(hf_cfg).eval()

    jcfg = Qwen3NextConfig.from_hf_config(hf_cfg)
    jmodel = Qwen3NextForCausalLM(jcfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    load_hf_qwen3_next_state_dict(jmodel, hf.state_dict(), param_dtype=jnp.float32)

    ids = np.random.default_rng(0).integers(0, hf_cfg.vocab_size, size=(2, 9)).astype(np.int64)
    with torch.no_grad():
        ref = hf(torch.from_numpy(ids)).logits.numpy()
    got = np.asarray(jmodel(jnp.asarray(ids.astype(np.int32))))

    np.testing.assert_allclose(got, ref, atol=2e-3, rtol=2e-3)


def test_worker_detects_qwen3_next_and_routes_refit():
    from dockyard_rl.models.jax.policy_worker import JaxPolicyWorkerImpl

    hf_cfg = _hf_cfg(mlp_only_layers=[])
    jcfg = Qwen3NextConfig.from_hf_config(hf_cfg)
    model = Qwen3NextForCausalLM(jcfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    worker = JaxPolicyWorkerImpl(model, init_optimizer=False)

    assert worker._is_qwen3_next is True
    assert worker._is_moe is True  # routes structurally through the MoE substrate
    # refit routes to the qwen3_next name-map (the conv special is the tell it's not the J8 path)
    info = worker.prepare_refit_info()
    assert info is not None and any(".conv1d.weight" in k for k in info)

    # logprob path runs end-to-end through the hybrid model (shape + zero-col convention)
    ids = np.random.default_rng(0).integers(0, hf_cfg.vocab_size, size=(2, 7)).astype(np.int64)
    out = worker.get_logprobs({"input_ids": torch.from_numpy(ids)})
    assert tuple(out["logprobs"].shape) == (2, 7)
    assert np.allclose(out["logprobs"].numpy()[:, 0], 0.0)


def test_loader_consumes_all_weights_strict():
    # strict=True (default) raises on any unconsumed HF weight -> a clean strict
    # load is itself the name-map completeness check.
    hf_cfg = _hf_cfg(mlp_only_layers=[1])
    hf = HFModel(hf_cfg).eval()
    jcfg = Qwen3NextConfig.from_hf_config(hf_cfg)
    jmodel = Qwen3NextForCausalLM(jcfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    load_hf_qwen3_next_state_dict(jmodel, hf.state_dict(), strict=True)  # must not raise
