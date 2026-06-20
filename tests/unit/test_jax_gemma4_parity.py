"""Gemma 4 (C1): CPU parity of the JAX Gemma-specific dense primitives vs HF.

Covers ``Gemma4TextConfig.from_hf_config`` field extraction and the C1 primitives
(``Gemma4RMSNorm`` incl. ``with_scale=False``, ``Gemma4MLP`` incl. the double-wide
variant, ``Gemma4ScaledWordEmbedding``) against ``transformers.models.gemma4`` on
CPU float32. Attention / KV-sharing / PLE / MoE / assembly land in C2-C6.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402
torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.gemma4")

from transformers.models.gemma4.configuration_gemma4 import (  # noqa: E402
    Gemma4TextConfig as HFGemma4TextConfig,
)
from transformers.models.gemma4.modeling_gemma4 import (  # noqa: E402
    Gemma4ForCausalLM as HFGemma4ForCausalLM,
    Gemma4RMSNorm as HFGemma4RMSNorm,
    Gemma4TextAttention as HFGemma4TextAttention,
    Gemma4TextExperts as HFGemma4TextExperts,
    Gemma4TextMLP as HFGemma4TextMLP,
    Gemma4TextModel as HFGemma4TextModel,
    Gemma4TextRotaryEmbedding as HFGemma4TextRotaryEmbedding,
    Gemma4TextRouter as HFGemma4TextRouter,
    Gemma4TextScaledWordEmbedding as HFScaledWordEmbedding,
)

from dockyard_rl.models.jax.models.gemma4 import (  # noqa: E402
    Gemma4Attention,
    Gemma4Experts,
    Gemma4ForCausalLM,
    Gemma4MLP,
    Gemma4PerLayerEmbeddings,
    Gemma4RMSNorm,
    Gemma4Router,
    Gemma4ScaledWordEmbedding,
    Gemma4TextConfig,
    gemma4_inv_freq,
    gemma4_rope_cos_sin,
    gemma4_sliding_bias,
    load_hf_gemma4_state_dict,
)


def _causal_mask_np(s: int) -> np.ndarray:
    i, j = np.arange(s)[:, None], np.arange(s)[None, :]
    m = np.where(j <= i, 0.0, np.finfo(np.float32).min).astype(np.float32)
    return m[None, None]


def _sliding_mask_np(s: int, w: int) -> np.ndarray:
    i, j = np.arange(s)[:, None], np.arange(s)[None, :]
    allowed = (j <= i) & (i - j < w)
    m = np.where(allowed, 0.0, np.finfo(np.float32).min).astype(np.float32)
    return m[None, None]


def _hf_text_config(**overrides):
    base = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_activation="gelu_pytorch_tanh",
        num_kv_shared_layers=0,
    )
    base.update(overrides)
    return HFGemma4TextConfig(**base)


def _t2j(t: torch.Tensor) -> jnp.ndarray:
    return jnp.asarray(t.detach().numpy())


# ── config extraction ───────────────────────────────────────────────


def test_from_hf_config_extracts_fields():
    hf = _hf_text_config(
        num_hidden_layers=12,
        global_head_dim=24,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        num_kv_shared_layers=3,
        sliding_window=64,
        final_logit_softcapping=30.0,
        enable_moe_block=True,
        num_experts=8,
        top_k_experts=2,
        moe_intermediate_size=40,
        use_double_wide_mlp=True,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=64,
    )
    cfg = Gemma4TextConfig.from_hf_config(hf)

    assert cfg.num_hidden_layers == 12
    assert cfg.global_head_dim == 24
    assert cfg.num_global_key_value_heads == 1
    assert cfg.attention_k_eq_v is True
    assert cfg.num_kv_shared_layers == 3
    assert cfg.final_logit_softcapping == 30.0
    assert cfg.enable_moe_block and cfg.num_experts == 8 and cfg.top_k_experts == 2
    assert cfg.use_double_wide_mlp is True
    # layer_types resolved by HF __post_init__: 5:1 sliding:full, last forced full.
    assert tuple(cfg.layer_types) == tuple(hf.layer_types)
    assert cfg.layer_types[-1] == "full_attention"
    # Per-layer-type RoPE: sliding=default theta=10k, full=proportional theta=1e6.
    assert cfg.sliding_rope_theta == 10_000.0
    assert cfg.global_rope_theta == 1_000_000.0
    assert cfg.global_partial_rotary_factor == 0.25
    # head_dim_for: sliding -> head_dim, full -> global_head_dim.
    full_idx = cfg.layer_types.index("full_attention")
    slide_idx = cfg.layer_types.index("sliding_attention")
    assert cfg.head_dim_for(full_idx) == 24
    assert cfg.head_dim_for(slide_idx) == 8
    # KV-shared layers are the last num_kv_shared_layers.
    assert cfg.is_kv_shared_layer(11) and not cfg.is_kv_shared_layer(8)
    # embed_scale = sqrt(hidden_size).
    np.testing.assert_allclose(cfg.embed_scale, 16.0**0.5, rtol=1e-6)


# ── Gemma4RMSNorm ───────────────────────────────────────────────────


@pytest.mark.parametrize("with_scale", [True, False])
def test_rmsnorm_parity(with_scale):
    dim = 16
    torch.manual_seed(0)
    hf = HFGemma4RMSNorm(dim, eps=1e-6, with_scale=with_scale)
    if with_scale:
        hf.weight.data = torch.randn(dim)  # ones-init would hide a scale bug
    jx = Gemma4RMSNorm(dim, eps=1e-6, with_scale=with_scale)
    if with_scale:
        jx.weight[...] = _t2j(hf.weight)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 5, dim)).astype(np.float32)
    out_hf = hf(torch.from_numpy(x)).detach().numpy()
    out_jx = np.asarray(jx(jnp.asarray(x)))
    np.testing.assert_allclose(out_jx, out_hf, atol=1e-5, rtol=1e-5)


# ── Gemma4MLP (incl. double-wide) ───────────────────────────────────


@pytest.mark.parametrize(
    "layer_idx,overrides",
    [
        (0, {}),                                                        # normal width
        (1, {"use_double_wide_mlp": True, "num_kv_shared_layers": 1}),  # double-wide (shared layer)
    ],
)
def test_mlp_parity(layer_idx, overrides):
    hf_cfg = _hf_text_config(**overrides)
    torch.manual_seed(0)
    hf = HFGemma4TextMLP(hf_cfg, layer_idx=layer_idx)
    for p in hf.parameters():
        with torch.no_grad():
            p.copy_(torch.randn_like(p))

    jcfg = Gemma4TextConfig.from_hf_config(hf_cfg)
    jx = Gemma4MLP(jcfg, jcfg.mlp_intermediate_size(layer_idx), rngs=nnx_rngs(), param_dtype=jnp.float32)
    # HF nn.Linear weight is (out, in); the JAX Linear kernel is (in, out).
    jx.gate_proj.kernel[...] = _t2j(hf.gate_proj.weight).T
    jx.up_proj.kernel[...] = _t2j(hf.up_proj.weight).T
    jx.down_proj.kernel[...] = _t2j(hf.down_proj.weight).T

    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 4, hf_cfg.hidden_size)).astype(np.float32)
    out_hf = hf(torch.from_numpy(x)).detach().numpy()
    out_jx = np.asarray(jx(jnp.asarray(x)))
    np.testing.assert_allclose(out_jx, out_hf, atol=2e-4, rtol=2e-4)
    # double-wide doubles the intermediate dim on the shared layer.
    expected = hf_cfg.intermediate_size * (2 if overrides.get("use_double_wide_mlp") else 1)
    assert jcfg.mlp_intermediate_size(layer_idx) == expected


# ── Gemma4ScaledWordEmbedding ───────────────────────────────────────


def test_scaled_word_embedding_parity():
    num, dim, scale = 64, 16, 4.0
    torch.manual_seed(0)
    hf = HFScaledWordEmbedding(num, dim, padding_idx=0, embed_scale=scale)
    with torch.no_grad():
        hf.weight.copy_(torch.randn(num, dim))

    jx = Gemma4ScaledWordEmbedding(num, dim, scale, rngs=nnx_rngs(), param_dtype=jnp.float32)
    jx.embedding[...] = _t2j(hf.weight)

    ids = np.random.default_rng(2).integers(0, num, size=(3, 7)).astype(np.int64)
    out_hf = hf(torch.from_numpy(ids)).detach().numpy()
    out_jx = np.asarray(jx(jnp.asarray(ids)))
    np.testing.assert_allclose(out_jx, out_hf, atol=1e-5, rtol=1e-5)


# ── C4: per-layer input embeddings (PLE) ────────────────────────────


def test_ple_parity():
    hf_cfg = _hf_text_config(
        num_hidden_layers=4, hidden_size=16, vocab_size=64,
        hidden_size_per_layer_input=8, vocab_size_per_layer_input=64,
    )
    torch.manual_seed(0)
    hf_model = HFGemma4TextModel(hf_cfg).eval()
    with torch.no_grad():
        hf_model.embed_tokens_per_layer.weight.copy_(torch.randn_like(hf_model.embed_tokens_per_layer.weight))
        hf_model.per_layer_model_projection.weight.copy_(torch.randn_like(hf_model.per_layer_model_projection.weight) * 0.1)
        hf_model.per_layer_projection_norm.weight.copy_(torch.randn_like(hf_model.per_layer_projection_norm.weight))
        hf_model.embed_tokens.weight.copy_(torch.randn_like(hf_model.embed_tokens.weight))

    ids = np.random.default_rng(4).integers(0, hf_cfg.vocab_size, size=(2, 5)).astype(np.int64)
    inputs_embeds = hf_model.embed_tokens(torch.from_numpy(ids))
    token_ple = hf_model.get_per_layer_inputs(torch.from_numpy(ids), inputs_embeds)
    ple_hf = hf_model.project_per_layer_inputs(inputs_embeds, token_ple).detach().numpy()

    jcfg = Gemma4TextConfig.from_hf_config(hf_cfg)
    jple = Gemma4PerLayerEmbeddings(jcfg, rngs=nnx_rngs(), param_dtype=jnp.float32)
    jple.embed_tokens_per_layer.embedding[...] = _t2j(hf_model.embed_tokens_per_layer.weight)
    jple.per_layer_model_projection.kernel[...] = _t2j(hf_model.per_layer_model_projection.weight).T
    jple.per_layer_projection_norm.weight[...] = _t2j(hf_model.per_layer_projection_norm.weight)

    out = jple(jnp.asarray(ids), jnp.asarray(inputs_embeds.detach().numpy()))
    np.testing.assert_allclose(np.asarray(out), ple_hf, atol=2e-4, rtol=2e-4)


# ── C6: full forward-logits parity (assembly + loader) ──────────────


def _full_cfg(**ov):
    base = dict(
        num_hidden_layers=8, hidden_size=16, intermediate_size=32,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        global_head_dim=16, num_global_key_value_heads=2,
        num_kv_shared_layers=2, attention_k_eq_v=True,
        hidden_size_per_layer_input=8, vocab_size_per_layer_input=64, vocab_size=64,
        final_logit_softcapping=30.0, sliding_window=4, max_position_embeddings=64,
    )
    base.update(ov)
    return _hf_text_config(**base)


@pytest.mark.parametrize(
    "moe", [False, True], ids=["dense_kvshare_ple_softcap", "with_moe"]
)
def test_forward_logits_parity(moe):
    extra = dict(enable_moe_block=True, num_experts=6, top_k_experts=2, moe_intermediate_size=24) if moe else {}
    hf_cfg = _full_cfg(**extra)
    hf_cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    hf = HFGemma4ForCausalLM(hf_cfg).eval()
    with torch.no_grad():
        for p in hf.parameters():
            p.copy_(torch.randn_like(p) * 0.1)

    ids = np.random.default_rng(7).integers(0, hf_cfg.vocab_size, size=(2, 9)).astype(np.int64)
    with torch.no_grad():
        logits_hf = hf(input_ids=torch.from_numpy(ids), use_cache=False).logits.numpy()

    jcfg = Gemma4TextConfig.from_hf_config(hf_cfg)
    jm = Gemma4ForCausalLM(jcfg, rngs=nnx_rngs(), param_dtype=jnp.float32)
    load_hf_gemma4_state_dict(jm, hf.state_dict())
    logits_jx = np.asarray(jm(jnp.asarray(ids)))
    np.testing.assert_allclose(logits_jx, logits_hf, atol=1e-3, rtol=1e-3)


# ── C5: MoE router + grouped experts ────────────────────────────────


def test_router_parity():
    hf_cfg = _hf_text_config(
        hidden_size=16, num_experts=8, top_k_experts=2,
        moe_intermediate_size=32, enable_moe_block=True,
    )
    torch.manual_seed(0)
    hf_router = HFGemma4TextRouter(hf_cfg)
    with torch.no_grad():
        hf_router.proj.weight.copy_(torch.randn_like(hf_router.proj.weight) * 0.1)
        hf_router.scale.copy_(torch.randn_like(hf_router.scale))
        hf_router.per_expert_scale.copy_(torch.randn_like(hf_router.per_expert_scale))

    t, d = 10, 16
    h = (np.random.default_rng(5).standard_normal((t, d)) * 0.5).astype(np.float32)
    _, top_w_hf, top_i_hf = hf_router(torch.from_numpy(h))

    jcfg = Gemma4TextConfig.from_hf_config(hf_cfg)
    jr = Gemma4Router(jcfg, rngs=nnx_rngs(), param_dtype=jnp.float32)
    jr.proj.kernel[...] = _t2j(hf_router.proj.weight).T
    jr.scale[...] = _t2j(hf_router.scale)
    jr.per_expert_scale[...] = _t2j(hf_router.per_expert_scale)

    top_w_jx, top_i_jx = jr(jnp.asarray(h))
    np.testing.assert_array_equal(np.asarray(top_i_jx), top_i_hf.detach().numpy())
    np.testing.assert_allclose(np.asarray(top_w_jx), top_w_hf.detach().numpy(), atol=2e-4, rtol=2e-4)


def test_experts_parity():
    hf_cfg = _hf_text_config(
        hidden_size=16, num_experts=6, top_k_experts=2,
        moe_intermediate_size=24, enable_moe_block=True,
    )
    torch.manual_seed(0)
    hf_exp = HFGemma4TextExperts(hf_cfg)
    with torch.no_grad():
        hf_exp.gate_up_proj.copy_(torch.randn_like(hf_exp.gate_up_proj) * 0.1)
        hf_exp.down_proj.copy_(torch.randn_like(hf_exp.down_proj) * 0.1)

    t, d, k, e = 12, 16, 2, 6
    rng = np.random.default_rng(6)
    h = (rng.standard_normal((t, d)) * 0.5).astype(np.float32)
    top_i = np.stack([rng.choice(e, size=k, replace=False) for _ in range(t)]).astype(np.int64)
    top_w = rng.random((t, k)).astype(np.float32)
    out_hf = hf_exp(torch.from_numpy(h), torch.from_numpy(top_i), torch.from_numpy(top_w)).detach().numpy()

    jcfg = Gemma4TextConfig.from_hf_config(hf_cfg)
    je = Gemma4Experts(jcfg, rngs=nnx_rngs(), param_dtype=jnp.float32)
    je.gate_up_proj[...] = _t2j(hf_exp.gate_up_proj)
    je.down_proj[...] = _t2j(hf_exp.down_proj)

    out_jx = je(jnp.asarray(h)[None], jnp.asarray(top_w)[None], jnp.asarray(top_i)[None])[0]
    np.testing.assert_allclose(np.asarray(out_jx), out_hf, atol=2e-4, rtol=2e-4)


# ── C2: per-layer-type RoPE + attention ─────────────────────────────


def _rope_cfg():
    return _hf_text_config(
        num_hidden_layers=6,
        head_dim=8,
        global_head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_rope_parity(layer_type):
    hf_cfg = _rope_cfg()
    hf_rotary = HFGemma4TextRotaryEmbedding(hf_cfg)
    jcfg = Gemma4TextConfig.from_hf_config(hf_cfg)

    b, s = 2, 7
    pos = np.tile(np.arange(s), (b, 1)).astype(np.int64)
    cos_hf, sin_hf = hf_rotary(torch.zeros(b, s, 1), torch.from_numpy(pos), layer_type=layer_type)

    inv = gemma4_inv_freq(jcfg, layer_type)
    cos_jx, sin_jx = gemma4_rope_cos_sin(jnp.asarray(pos), inv)
    np.testing.assert_allclose(np.asarray(cos_jx), cos_hf.detach().numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(sin_jx), sin_hf.detach().numpy(), atol=1e-5, rtol=1e-5)
    # Proportional partial-rotary: global layers rotate only int(0.25*16//2)=2 freqs,
    # the remaining head_dim//2-2 = 6 inv_freq entries are zero (NoPE / identity).
    if layer_type == "full_attention":
        assert int(np.asarray(inv).shape[0]) == 8  # global_head_dim // 2
        assert np.count_nonzero(np.asarray(inv)) == 2


def test_sliding_bias_matches_reference():
    s, w = 9, 4
    np.testing.assert_array_equal(np.asarray(gemma4_sliding_bias(s, w, jnp.float32)), _sliding_mask_np(s, w))


@pytest.mark.parametrize(
    "layer_type,k_eq_v",
    [
        ("sliding_attention", False),
        ("full_attention", False),
        ("full_attention", True),  # attention_k_eq_v: V reuses the K projection
    ],
)
def test_attention_parity(layer_type, k_eq_v):
    hf_cfg = _hf_text_config(
        num_hidden_layers=6,
        head_dim=8,
        global_head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        attention_k_eq_v=k_eq_v,
        num_kv_shared_layers=0,
        max_position_embeddings=64,
    )
    hf_cfg._attn_implementation = "eager"
    jcfg = Gemma4TextConfig.from_hf_config(hf_cfg)
    layer_idx = list(jcfg.layer_types).index(layer_type)
    uses_k_eq_v = k_eq_v and layer_type == "full_attention"

    hf_rotary = HFGemma4TextRotaryEmbedding(hf_cfg)
    torch.manual_seed(0)
    hf_attn = HFGemma4TextAttention(hf_cfg, layer_idx)
    for p in hf_attn.parameters():
        with torch.no_grad():
            p.copy_(torch.randn_like(p) * 0.1)

    b, s, h = 2, 7, hf_cfg.hidden_size
    pos = np.tile(np.arange(s), (b, 1)).astype(np.int64)
    x = (np.random.default_rng(3).standard_normal((b, s, h)) * 0.5).astype(np.float32)
    cos, sin = hf_rotary(torch.from_numpy(x), torch.from_numpy(pos), layer_type=layer_type)

    mask_np = _sliding_mask_np(s, jcfg.sliding_window) if layer_type == "sliding_attention" else _causal_mask_np(s)

    out_hf, _ = hf_attn(
        torch.from_numpy(x), position_embeddings=(cos, sin),
        attention_mask=torch.from_numpy(mask_np), shared_kv_states={},
    )

    jx = Gemma4Attention(jcfg, layer_idx, rngs=nnx_rngs(), param_dtype=jnp.float32)
    jx.q_proj.kernel[...] = _t2j(hf_attn.q_proj.weight).T
    jx.k_proj.kernel[...] = _t2j(hf_attn.k_proj.weight).T
    jx.o_proj.kernel[...] = _t2j(hf_attn.o_proj.weight).T
    jx.q_norm.weight[...] = _t2j(hf_attn.q_norm.weight)
    jx.k_norm.weight[...] = _t2j(hf_attn.k_norm.weight)
    if not uses_k_eq_v:
        jx.v_proj.kernel[...] = _t2j(hf_attn.v_proj.weight).T
    else:
        assert jx.v_proj is None and hf_attn.v_proj is None

    out_jx, _ = jx(jnp.asarray(x), jnp.asarray(cos.detach().numpy()), jnp.asarray(sin.detach().numpy()), jnp.asarray(mask_np))
    np.testing.assert_allclose(np.asarray(out_jx), out_hf.detach().numpy(), atol=2e-4, rtol=2e-4)


def nnx_rngs():
    from flax import nnx

    return nnx.Rngs(params=0)
