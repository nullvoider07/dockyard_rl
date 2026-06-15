"""CPU forward-logits parity: JAX (Flax NNX) Qwen3 dense vs HF transformers.

The key J1 validation lever: both backends run on CPU in float32, so a tiny
random-init Qwen3 must produce identical logits once the HF weights are loaded
into the NNX tree via the name-map loader. Exercises GQA (n_kv < n_heads),
the per-head q_norm/k_norm, explicit head_dim, and both tied and untied
lm_head.
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
from dockyard_rl.models.jax.weights import load_hf_state_dict


def _build_hf(tie: bool) -> tuple[HFQwen3ForCausalLM, HFQwen3Config]:
    # rope_theta / attn_implementation are accepted by HF Qwen3Config via **kwargs
    # (verified at runtime); the typed stub does not enumerate them, hence the ignores.
    cfg = HFQwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,   # GQA: 2 kv groups
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,  # pyright: ignore[reportCallIssue]
        attention_bias=False,
        tie_word_embeddings=tie,
        attn_implementation="eager",  # pyright: ignore[reportCallIssue]
    )
    torch.manual_seed(0)
    model = HFQwen3ForCausalLM(cfg).to(torch.float32).eval()
    return model, cfg


@pytest.mark.parametrize("tie", [False, True])
def test_qwen3_forward_logits_parity(tie: bool) -> None:
    hf_model, hf_cfg = _build_hf(tie)

    jax_cfg = Qwen3Config.from_hf_config(hf_cfg)
    jax_model = Qwen3ForCausalLM(jax_cfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    load_hf_state_dict(jax_model, hf_model.state_dict(), param_dtype=jnp.float32)

    rng = np.random.default_rng(0)
    ids = rng.integers(0, hf_cfg.vocab_size, size=(2, 9)).astype(np.int64)

    with torch.no_grad():
        hf_logits = hf_model(input_ids=torch.from_numpy(ids)).logits.float().numpy()
    jax_logits = np.asarray(jax_model(jnp.asarray(ids)))

    assert jax_logits.shape == hf_logits.shape == (2, 9, hf_cfg.vocab_size)
    np.testing.assert_allclose(jax_logits, hf_logits, atol=2e-4, rtol=2e-4)


def test_tied_embeddings_share_param() -> None:
    # A tied model must have NO separate lm_head param (logits reuse the
    # embedding), so the tie survives training instead of diverging.
    cfg = Qwen3Config.from_hf_config(_build_hf(tie=True)[1])
    model = Qwen3ForCausalLM(cfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    paths = [tuple(p) for p, _ in nnx.to_flat_state(nnx.state(model))]
    assert not any("lm_head" in p for p in paths)
    # exactly one (vocab, hidden) matrix — the shared embedding
    vocab_hidden = [p for p, v in nnx.to_flat_state(nnx.state(model))
                    if v[...].shape == (cfg.vocab_size, cfg.hidden_size)]
    assert len(vocab_hidden) == 1


def test_name_map_roundtrip() -> None:
    from dockyard_rl.models.jax.weights import hf_name_to_nnx_path, nnx_path_to_hf_name

    names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.q_norm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for name in names:
        path = hf_name_to_nnx_path(name, num_layers=2)
        assert nnx_path_to_hf_name(path) == name
