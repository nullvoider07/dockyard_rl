"""Tests for model-agnostic MoE detection (pure, no GPU)."""

from __future__ import annotations

from dockyard_rl.models.dtensor.moe.detect import is_moe_state_dict


def test_qwen3_moe_keys_detected():
    keys = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
    ]
    assert is_moe_state_dict(keys)


def test_native_grouped_experts_detected():
    assert is_moe_state_dict(["model.layers.0.mlp.experts.w1_EFD"])


def test_mixtral_block_sparse_detected():
    assert is_moe_state_dict(
        ["model.layers.0.block_sparse_moe.experts.3.w1.weight"]
    )


def test_dense_model_not_detected():
    keys = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",  # dense SwiGLU gate, not a router
        "model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
    ]
    assert not is_moe_state_dict(keys)


def test_empty():
    assert not is_moe_state_dict([])
