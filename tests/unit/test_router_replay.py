"""Tests for MoE router-replay: the router hook + set-and-consume controller +
config validation (pure, CPU).

The router replay path is tested directly on TokenChoiceTopKRouter (forced ids,
sentinel fallback, gating re-gather, gradient, shape validation). The controller
is tested with MoEBlocks built on a stub experts module so MoEBlock.forward runs
on CPU (the real grouped-GEMM experts are CUDA-only, HV-8). No GPU.
"""

from __future__ import annotations

from typing import Optional, cast

import torch
from torch import nn

from dockyard_rl.models.dtensor.moe.block import MoEBlock
from dockyard_rl.models.dtensor.moe.router import (
    MISSING_ROUTE_SENTINEL,
    TokenChoiceTopKRouter,
)
from dockyard_rl.models.dtensor.moe.router_replay import (
    bind_router_replay,
    count_moe_blocks,
    iter_moe_blocks,
    resolve_router_replay_enabled,
    validate_router_replay_config,
)
from dockyard_rl.models.generation.vllm import router_capture


# --------------------------------------------------------------------------- #
# Router replay hook
# --------------------------------------------------------------------------- #
def test_default_path_unchanged_when_no_replay():
    r = TokenChoiceTopKRouter(dim=8, num_experts=6, top_k=2)
    x = torch.randn(2, 3, 8)
    s0, ids0, sc0 = r(x)
    s1, ids1, sc1 = r(x, None, None)
    assert torch.equal(ids0, ids1)
    assert torch.equal(s0, s1)
    assert torch.equal(sc0, sc1)


def test_replay_forces_recorded_ids():
    r = TokenChoiceTopKRouter(dim=8, num_experts=6, top_k=2)
    x = torch.randn(2, 3, 8)
    _, computed_ids, _ = r(x)
    # A replay route distinct from what the router would compute.
    replay = torch.full((2, 3, 2), 5, dtype=torch.int32)
    replay[..., 0] = 1
    topk_scores, ids, scores = r(x, None, replay)
    assert torch.equal(ids, replay.to(torch.long))
    assert not torch.equal(ids, computed_ids)  # genuinely overridden
    # Gating weights are re-gathered from THIS model's scores at the forced ids.
    expected = scores.gather(-1, replay.to(torch.long))
    assert torch.allclose(topk_scores, expected)


def test_replay_sentinel_falls_back_to_computed():
    r = TokenChoiceTopKRouter(dim=8, num_experts=6, top_k=2)
    x = torch.randn(1, 4, 8)
    _, computed_ids, _ = r(x)
    replay = torch.full((1, 4, 2), 3, dtype=torch.int32)
    # Mark token positions 1 and 2 as missing (whole-K sentinel rows).
    replay[0, 1, :] = MISSING_ROUTE_SENTINEL
    replay[0, 2, :] = MISSING_ROUTE_SENTINEL
    _, ids, _ = r(x, None, replay)
    # Valid positions (0, 3) forced to the recorded ids.
    assert ids[0, 0].tolist() == [3, 3]
    assert ids[0, 3].tolist() == [3, 3]
    # Sentinel positions (1, 2) fall back to the computed routing.
    assert torch.equal(ids[0, 1], computed_ids[0, 1])
    assert torch.equal(ids[0, 2], computed_ids[0, 2])


def test_replay_preserves_route_norm_and_scale():
    r = TokenChoiceTopKRouter(
        dim=8, num_experts=6, top_k=2, route_norm=True, route_scale=3.0
    )
    x = torch.randn(2, 3, 8)
    replay = torch.zeros(2, 3, 2, dtype=torch.int32)
    replay[..., 1] = 2
    topk_scores, ids, scores = r(x, None, replay)
    gathered = scores.gather(-1, ids)
    normed = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20) * 3.0
    assert torch.allclose(topk_scores, normed)


def test_replay_gating_gradient_flows_through_gate():
    r = TokenChoiceTopKRouter(dim=8, num_experts=6, top_k=2)
    x = torch.randn(2, 3, 8)
    replay = torch.zeros(2, 3, 2, dtype=torch.int32)
    replay[..., 1] = 4
    topk_scores, _, _ = r(x, None, replay)
    topk_scores.sum().backward()
    assert r.gate.weight.grad is not None
    assert torch.isfinite(r.gate.weight.grad).all()
    assert r.gate.weight.grad.abs().sum() > 0


def test_replay_wrong_topk_dim_raises():
    r = TokenChoiceTopKRouter(dim=8, num_experts=6, top_k=2)
    x = torch.randn(2, 3, 8)
    bad = torch.zeros(2, 3, 3, dtype=torch.int32)  # K=3 != top_k=2
    try:
        r(x, None, bad)
    except ValueError as e:
        assert "top_k" in str(e)
    else:
        raise AssertionError("expected ValueError on top_k mismatch")


def test_replay_wrong_batch_seq_raises():
    r = TokenChoiceTopKRouter(dim=8, num_experts=6, top_k=2)
    x = torch.randn(2, 3, 8)
    bad = torch.zeros(2, 5, 2, dtype=torch.int32)  # L=5 != 3
    try:
        r(x, None, bad)
    except ValueError as e:
        assert "match scores" in str(e)
    else:
        raise AssertionError("expected ValueError on batch/seq mismatch")


def test_replay_with_expert_bias_choice_forced_but_gating_unbiased():
    r = TokenChoiceTopKRouter(dim=8, num_experts=6, top_k=2)
    x = torch.randn(2, 3, 8)
    bias = torch.zeros(6)
    bias[5] = 100.0  # would dominate the CHOICE if it were used
    replay = torch.zeros(2, 3, 2, dtype=torch.int32)
    replay[..., 1] = 1
    topk_scores, ids, scores = r(x, bias, replay)
    # Replay forces ids regardless of the bias.
    assert torch.equal(ids, replay.to(torch.long))
    # Gating still from unbiased scores.
    assert torch.allclose(topk_scores, scores.gather(-1, ids))


# --------------------------------------------------------------------------- #
# Controller (set-and-consume) — MoEBlock with stub experts
# --------------------------------------------------------------------------- #
class _StubExperts(nn.Module):
    """Records the routing ids it receives; shape-faithful, no grouped-GEMM."""

    def __init__(self) -> None:
        super().__init__()
        self.last_ids: Optional[torch.Tensor] = None

    def forward(self, x_BLD, topk_scores_BLK, topk_expert_ids_BLK, *_):
        self.last_ids = topk_expert_ids_BLK
        return x_BLD * topk_scores_BLK.sum(dim=-1, keepdim=True)


def _moe_model(num_layers: int, num_experts: int = 6, top_k: int = 2) -> nn.Module:
    torch.manual_seed(0)
    model = nn.Module()
    blocks = nn.ModuleList()
    for _ in range(num_layers):
        router = TokenChoiceTopKRouter(dim=8, num_experts=num_experts, top_k=top_k)
        blocks.append(
            MoEBlock(router, _StubExperts(), num_experts=num_experts)  # type: ignore[arg-type]
        )
    model.blocks = blocks  # type: ignore[assignment]
    return model


def test_iter_and_count_moe_blocks():
    model = _moe_model(3)
    assert count_moe_blocks(model) == 3
    assert len(iter_moe_blocks(model)) == 3
    dense = nn.Sequential(nn.Linear(8, 8))
    assert count_moe_blocks(dense) == 0


def test_bind_sets_slices_and_clears():
    model = _moe_model(2)
    blocks = iter_moe_blocks(model)
    # routed_experts [B=1, T=3, L_moe=2, K=2]
    routed = torch.zeros(1, 3, 2, 2, dtype=torch.int32)
    routed[:, :, 0, :] = 1  # layer 0 -> expert 1
    routed[:, :, 1, :] = 4  # layer 1 -> expert 4
    with bind_router_replay(model, routed):
        assert blocks[0]._replay_route_BLK is not None
        assert torch.equal(blocks[0]._replay_route_BLK, routed[:, :, 0, :])
        assert torch.equal(blocks[1]._replay_route_BLK, routed[:, :, 1, :])
    # Cleared on exit.
    assert blocks[0]._replay_route_BLK is None
    assert blocks[1]._replay_route_BLK is None


def test_bind_drives_block_forward_routing():
    model = _moe_model(2)
    blocks = iter_moe_blocks(model)
    x = torch.randn(1, 3, 8)
    routed = torch.zeros(1, 3, 2, 2, dtype=torch.int32)
    routed[:, :, 0, :] = torch.tensor([1, 2])
    routed[:, :, 1, :] = torch.tensor([3, 5])
    with bind_router_replay(model, routed):
        for blk in blocks:
            blk(x)
    ids0 = cast(torch.Tensor, blocks[0].experts.last_ids)
    ids1 = cast(torch.Tensor, blocks[1].experts.last_ids)
    assert ids0[0, 0].tolist() == [1, 2]
    assert ids1[0, 0].tolist() == [3, 5]


def test_bind_clears_even_on_exception():
    model = _moe_model(2)
    blocks = iter_moe_blocks(model)
    routed = torch.zeros(1, 3, 2, 2, dtype=torch.int32)
    try:
        with bind_router_replay(model, routed):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert all(b._replay_route_BLK is None for b in blocks)


def test_bind_layer_count_mismatch_raises():
    model = _moe_model(3)
    routed = torch.zeros(1, 3, 2, 2, dtype=torch.int32)  # L_moe=2 != 3 blocks
    try:
        with bind_router_replay(model, routed):
            pass
    except ValueError as e:
        assert "MoEBlock count" in str(e)
    else:
        raise AssertionError("expected ValueError on layer-count mismatch")


def test_bind_no_moe_raises():
    dense = nn.Sequential(nn.Linear(8, 8))
    routed = torch.zeros(1, 3, 1, 2, dtype=torch.int32)
    try:
        with bind_router_replay(dense, routed):
            pass
    except ValueError as e:
        assert "no MoEBlock" in str(e)
    else:
        raise AssertionError("expected ValueError on dense model")


def test_bind_non_4d_raises():
    model = _moe_model(1)
    bad = torch.zeros(1, 3, 2, dtype=torch.int32)  # 3D
    try:
        with bind_router_replay(model, bad):
            pass
    except ValueError as e:
        assert "[B, T, L_moe, K]" in str(e)
    else:
        raise AssertionError("expected ValueError on non-4D routing")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_resolve_router_replay_enabled():
    assert resolve_router_replay_enabled({"router_replay": {"enabled": True}}) is True
    assert resolve_router_replay_enabled({"router_replay": {"enabled": False}}) is False
    assert resolve_router_replay_enabled({}) is False
    assert resolve_router_replay_enabled({"router_replay": None}) is False

    class _Cfg:
        router_replay = {"enabled": True}

    assert resolve_router_replay_enabled(_Cfg()) is True


def test_validate_router_replay_config():
    moe = _moe_model(2)
    dense = nn.Sequential(nn.Linear(8, 8))
    # Enabled on MoE: ok.
    validate_router_replay_config(True, moe)
    # Disabled: always ok.
    validate_router_replay_config(False, dense)
    # Enabled on dense: raises.
    try:
        validate_router_replay_config(True, dense)
    except ValueError as e:
        assert "requires an MoE model" in str(e)
    else:
        raise AssertionError("expected ValueError on dense + enabled")


def test_sentinel_constant_does_not_drift():
    # The producer (capture) and consumer (router) must agree on the sentinel.
    assert MISSING_ROUTE_SENTINEL == router_capture.MISSING_ROUTE_SENTINEL
