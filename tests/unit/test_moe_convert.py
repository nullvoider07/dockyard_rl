"""Tests for HF per-expert -> fused conversion, incl. round-trip vs refit."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from dockyard_rl.models.dtensor.moe.convert import (
    fuse_hf_expert_state_dict,
    is_hf_per_expert_key,
)
from dockyard_rl.models.dtensor.moe.refit import iter_expanded_refit_tensors


def test_is_hf_per_expert_key():
    assert is_hf_per_expert_key("model.layers.0.mlp.experts.0.gate_proj.weight")
    assert is_hf_per_expert_key("model.layers.3.mlp.experts.7.down_proj.weight")


def test_is_not_hf_per_expert_key():
    assert not is_hf_per_expert_key("model.layers.0.mlp.gate.weight")
    assert not is_hf_per_expert_key("model.layers.0.mlp.experts.w1_EFD")
    assert not is_hf_per_expert_key("model.layers.0.self_attn.q_proj.weight")


def test_fuse_stacks_in_expert_order():
    base = "model.layers.0.mlp.experts"
    E, F, D = 3, 5, 4
    items = []
    expected_rows = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        rows = []
        for i in range(E):
            shape = (D, F) if proj == "down_proj" else (F, D)
            t = torch.randn(*shape)
            rows.append(t)
            items.append((f"{base}.{i}.{proj}.weight", t))
        expected_rows[proj] = rows

    out = fuse_hf_expert_state_dict(items)
    assert torch.equal(out[f"{base}.w1_EFD"], torch.stack(expected_rows["gate_proj"]))
    assert torch.equal(out[f"{base}.w3_EFD"], torch.stack(expected_rows["up_proj"]))
    assert torch.equal(out[f"{base}.w2_EDF"], torch.stack(expected_rows["down_proj"]))
    assert out[f"{base}.w1_EFD"].shape == (E, F, D)
    assert out[f"{base}.w2_EDF"].shape == (E, D, F)


def test_non_expert_keys_pass_through():
    attn = torch.zeros(2, 2)
    items = [
        ("model.layers.0.self_attn.q_proj.weight", attn),
        ("model.layers.0.mlp.gate.weight", torch.zeros(4, 8)),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.zeros(5, 8)),
        ("model.layers.0.mlp.experts.1.gate_proj.weight", torch.zeros(5, 8)),
    ]
    out = fuse_hf_expert_state_dict(items)
    assert out["model.layers.0.self_attn.q_proj.weight"] is attn
    assert "model.layers.0.mlp.gate.weight" in out
    assert "model.layers.0.mlp.experts.w1_EFD" in out


def test_non_contiguous_indices_rejected():
    base = "model.layers.0.mlp.experts"
    items = [
        (f"{base}.0.gate_proj.weight", torch.zeros(5, 4)),
        (f"{base}.2.gate_proj.weight", torch.zeros(5, 4)),  # missing index 1
    ]
    with pytest.raises(ValueError):
        fuse_hf_expert_state_dict(items)


def test_round_trip_fuse_then_expand_is_identity():
    # fuse (load side) then expand (refit side) must reproduce the original
    # per-expert HF tensors exactly.
    base = "model.layers.2.mlp.experts"
    E, F, D = 4, 6, 3
    original = {}
    for proj, shape in (
        ("gate_proj", (F, D)),
        ("up_proj", (F, D)),
        ("down_proj", (D, F)),
    ):
        for i in range(E):
            original[f"{base}.{i}.{proj}.weight"] = torch.randn(*shape)

    fused = fuse_hf_expert_state_dict(original.items())
    # fused has only the 3 stacked params
    assert set(fused) == {f"{base}.w1_EFD", f"{base}.w3_EFD", f"{base}.w2_EDF"}

    expanded = dict(
        iter_expanded_refit_tensors(fused.items(), materialize=lambda t: t)
    )
    assert set(expanded) == set(original)
    for k in original:
        assert torch.equal(cast(torch.Tensor, expanded[k]), original[k])
