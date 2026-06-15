"""Tests for EP-aware refit expansion (pure, no GPU).

Covers the native GroupedExperts -> per-expert HF name/shape/unbind logic. The
EP gather (``full_tensor``) is injected as ``materialize`` so the flat-map is
exercised without a device mesh.
"""

from __future__ import annotations

from typing import cast

import torch

from dockyard_rl.models.dtensor.moe.refit import (
    expand_grouped_expert_info,
    expand_grouped_expert_key,
    is_grouped_expert_key,
    iter_expanded_refit_tensors,
)


def test_is_grouped_expert_key():
    assert is_grouped_expert_key("model.layers.0.mlp.experts.w1_EFD")
    assert is_grouped_expert_key("model.layers.3.mlp.experts.w2_EDF")
    assert is_grouped_expert_key("model.layers.3.mlp.experts.w3_EFD")


def test_is_not_grouped_expert_key():
    assert not is_grouped_expert_key("model.layers.0.self_attn.q_proj.weight")
    assert not is_grouped_expert_key("model.layers.0.mlp.gate.weight")
    # Already-expanded HF per-expert key must not re-match.
    assert not is_grouped_expert_key(
        "model.layers.0.mlp.experts.0.gate_proj.weight"
    )


def test_expand_key_proj_mapping():
    base = "model.layers.2.mlp.experts"
    assert (
        expand_grouped_expert_key(f"{base}.w1_EFD", 5)
        == f"{base}.5.gate_proj.weight"
    )
    assert (
        expand_grouped_expert_key(f"{base}.w3_EFD", 5)
        == f"{base}.5.up_proj.weight"
    )
    assert (
        expand_grouped_expert_key(f"{base}.w2_EDF", 0)
        == f"{base}.0.down_proj.weight"
    )


def test_expand_info_shapes():
    # w1_EFD: (E, F, D) -> per-expert (F, D)
    info = expand_grouped_expert_info(
        "model.layers.0.mlp.experts.w1_EFD", (4, 16, 8)
    )
    assert len(info) == 4
    assert info[0] == ("model.layers.0.mlp.experts.0.gate_proj.weight", (16, 8))
    assert info[3] == ("model.layers.0.mlp.experts.3.gate_proj.weight", (16, 8))

    # w2_EDF: (E, D, F) -> per-expert (D, F)
    down = expand_grouped_expert_info(
        "model.layers.0.mlp.experts.w2_EDF", (4, 8, 16)
    )
    assert down[2] == ("model.layers.0.mlp.experts.2.down_proj.weight", (8, 16))


def test_iter_expands_experts_and_passes_through():
    E, F, D = 3, 5, 4
    w1 = torch.arange(E * F * D, dtype=torch.float32).reshape(E, F, D)
    attn = torch.zeros(4, 4)
    items = [
        ("model.layers.0.self_attn.q_proj.weight", attn),
        ("model.layers.0.mlp.experts.w1_EFD", w1),
    ]

    materialized_ids: list[int] = []

    def materialize(t):
        materialized_ids.append(id(t))
        return t

    out = dict(iter_expanded_refit_tensors(items, materialize))

    # Non-expert param passed through unchanged and NOT materialized.
    assert "model.layers.0.self_attn.q_proj.weight" in out
    assert out["model.layers.0.self_attn.q_proj.weight"] is attn
    assert id(attn) not in materialized_ids

    # Expert param expanded into E per-expert tensors, each == the unbound row.
    assert id(w1) in materialized_ids
    for i in range(E):
        key = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
        assert key in out
        val = cast(torch.Tensor, out[key])
        assert torch.equal(val, w1[i])
        assert val.shape == (F, D)


def test_iter_no_experts_is_identity():
    items = [
        ("model.layers.0.self_attn.q_proj.weight", torch.zeros(2, 2)),
        ("model.layers.0.mlp.gate.weight", torch.zeros(3, 2)),
    ]
    out = list(iter_expanded_refit_tensors(items, materialize=lambda t: t))
    assert [n for n, _ in out] == [n for n, _ in items]


def test_emitted_names_match_vllm_fusedmoe_contract():
    """Lock the per-expert names against vLLM's FusedMoE loader (HV-13, slice 3.3).

    vLLM 0.21/0.22 ``Qwen3MoeForCausalLM.get_expert_mapping`` calls
    ``make_expert_params_mapping(ckpt_gate_proj_name="gate_proj",
    ckpt_down_proj_name="down_proj", ckpt_up_proj_name="up_proj", ...)``, yielding
    per (expert i, shard) entries with ``weight_name = f"experts.{i}.{proj}."``
    and ``param_name = "experts.w13_"`` (gate/up) | ``"experts.w2_"`` (down). The
    loader matches an incoming weight by ``weight_name in name`` then does
    ``name.replace(weight_name, param_name)`` to land on the FusedMoE param
    (``...experts.w13_weight`` / ``...experts.w2_weight``). This replays that on
    our emitted names so a convention drift fails here, not silently in vLLM.
    Source verified: vllm/model_executor/layers/fused_moe/layer.py::make_expert_params_mapping
    + vllm/model_executor/models/qwen3_moe.py::load_weights.
    """
    base = "model.layers.0.mlp.experts"
    num_experts = 4

    def vllm_expert_mapping(i: int) -> list[tuple[str, str]]:
        # (param_name, weight_name) for the three shards of expert i.
        return [
            ("experts.w13_", f"experts.{i}.gate_proj."),  # shard w1 (gate)
            ("experts.w2_", f"experts.{i}.down_proj."),    # shard w2 (down)
            ("experts.w13_", f"experts.{i}.up_proj."),     # shard w3 (up)
        ]

    fused = ["w1_EFD", "w3_EFD", "w2_EDF"]
    expected_param = {
        "w1_EFD": "model.layers.0.mlp.experts.w13_weight",
        "w3_EFD": "model.layers.0.mlp.experts.w13_weight",
        "w2_EDF": "model.layers.0.mlp.experts.w2_weight",
    }
    for i in range(num_experts):
        mapping = vllm_expert_mapping(i)
        for suffix in fused:
            emitted = expand_grouped_expert_key(f"{base}.{suffix}", i)
            # Exactly one vLLM mapping entry substring-matches this emitted name.
            matches = [
                (param_name, weight_name)
                for (param_name, weight_name) in mapping
                if weight_name in emitted
            ]
            assert len(matches) == 1, (emitted, matches)
            param_name, weight_name = matches[0]
            assert emitted.replace(weight_name, param_name) == expected_param[suffix]
