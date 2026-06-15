"""J8d: Qwen3-MoE forward-logit parity, JAX (Flax NNX) vs HF transformers (CPU).

The whole MoE compute core (router softmax-topk, grouped-GEMM SwiGLU experts via
``ragged_dot``, score-after-experts combine) plus the fused-expert HF loader are
validated end-to-end here: a tiny random Qwen3-MoE must produce logits identical
to ``transformers`` once its weights are loaded. Exercises GQA, the per-head
q_norm/k_norm, the fused ``gate_up_proj`` split, ``norm_topk_prob``, and tied/untied.
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
from dockyard_rl.models.jax.moe.weights import load_hf_qwen3_moe_state_dict


def _build_hf(tie: bool, norm_topk_prob: bool, mlp_only_layers=()):
    cfg = HFConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,  # pyright: ignore[reportCallIssue]
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,  # pyright: ignore[reportCallIssue]
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        decoder_sparse_step=1,
        mlp_only_layers=list(mlp_only_layers),
        norm_topk_prob=norm_topk_prob,
        tie_word_embeddings=tie,
        attn_implementation="eager",  # pyright: ignore[reportCallIssue]
    )
    torch.manual_seed(0)
    return HFModel(cfg).to(torch.float32).eval(), cfg


@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize("norm_topk_prob", [False, True])
def test_qwen3_moe_forward_parity(tie, norm_topk_prob):
    hf_model, hf_cfg = _build_hf(tie, norm_topk_prob)
    jcfg = Qwen3MoeConfig.from_hf_config(hf_cfg)
    jm = Qwen3MoeForCausalLM(jcfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    load_hf_qwen3_moe_state_dict(jm, hf_model.state_dict(), param_dtype=jnp.float32)

    ids = np.random.default_rng(0).integers(0, hf_cfg.vocab_size, size=(2, 9)).astype(np.int64)
    with torch.no_grad():
        hf_logits = hf_model(input_ids=torch.from_numpy(ids)).logits.float().numpy()
    jax_logits = np.asarray(jm(jnp.asarray(ids)))

    assert jax_logits.shape == hf_logits.shape == (2, 9, hf_cfg.vocab_size)
    np.testing.assert_allclose(jax_logits, hf_logits, atol=2e-4, rtol=2e-4)


def test_qwen3_moe_mixed_dense_sparse_layers():
    # mlp_only_layers makes layer 0 a dense Qwen3MoeMLP; layer 1 stays sparse.
    hf_model, hf_cfg = _build_hf(tie=False, norm_topk_prob=False, mlp_only_layers=(0,))
    jcfg = Qwen3MoeConfig.from_hf_config(hf_cfg)
    assert jcfg.is_sparse_layer(0) is False and jcfg.is_sparse_layer(1) is True
    jm = Qwen3MoeForCausalLM(jcfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    load_hf_qwen3_moe_state_dict(jm, hf_model.state_dict(), param_dtype=jnp.float32)

    ids = np.random.default_rng(1).integers(0, hf_cfg.vocab_size, size=(1, 7)).astype(np.int64)
    with torch.no_grad():
        hf_logits = hf_model(input_ids=torch.from_numpy(ids)).logits.float().numpy()
    jax_logits = np.asarray(jm(jnp.asarray(ids)))
    np.testing.assert_allclose(jax_logits, hf_logits, atol=2e-4, rtol=2e-4)
