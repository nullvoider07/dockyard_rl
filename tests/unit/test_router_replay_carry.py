"""Tests for the MoE router-replay rollout->train carry (#2908, pure CPU).

The capture (vLLM) and live multi-rank replay are GPU-deferred (HV-47), but the
data carry is fully CPU-checkable: a per-assistant-message ``routed_experts``
slice rides the same message_log -> flat-batch collation as
``generation_logprobs`` and must arrive as a ``[B, T, L, K]`` column that the
trainer-side ``bind_router_replay`` controller can bind. These tests drive the
real collation (``batched_message_log_to_flat_message``) and binder, asserting
the layout, the sentinel padding, and the end-to-end round-trip.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from dockyard_rl.data.llm_message_utils import (
    backfill_routed_experts,
    batched_message_log_to_flat_message,
)
from dockyard_rl.models.dtensor.moe.block import MoEBlock
from dockyard_rl.models.dtensor.moe.router import TokenChoiceTopKRouter
from dockyard_rl.models.dtensor.moe.router_replay import bind_router_replay
from dockyard_rl.models.generation.vllm.router_capture import MISSING_ROUTE_SENTINEL

_L = 3   # MoE layers
_K = 2   # top-k


class _StubExperts(nn.Module):
    """Records the routing ids it receives; shape-faithful, no grouped-GEMM."""

    def __init__(self) -> None:
        super().__init__()
        self.last_ids: Optional[torch.Tensor] = None

    def forward(self, x_BLD, topk_scores_BLK, topk_expert_ids_BLK, *_):
        self.last_ids = topk_expert_ids_BLK
        return x_BLD * topk_scores_BLK.sum(dim=-1, keepdim=True)


def _moe_model(num_layers: int) -> nn.Module:
    torch.manual_seed(0)
    model = nn.Module()
    blocks = nn.ModuleList()
    for _ in range(num_layers):
        router = TokenChoiceTopKRouter(dim=8, num_experts=6, top_k=_K)
        blocks.append(MoEBlock(router, _StubExperts(), num_experts=6))  # type: ignore[arg-type]
    model.blocks = blocks  # type: ignore[assignment]
    return model


def _routes(msg_len: int, start: int) -> torch.Tensor:
    """Distinct per-position routes [msg_len, L, K] so placement is verifiable."""
    base = torch.arange(msg_len, dtype=torch.int32).view(msg_len, 1, 1)
    grid = torch.arange(_L * _K, dtype=torch.int32).view(1, _L, _K)
    return base * 100 + grid + start


def _message_log(msg_len: int, start: int) -> list:
    return [
        {
            "role": "assistant",
            "token_ids": torch.arange(msg_len, dtype=torch.int64),
            "generation_logprobs": torch.zeros(msg_len, dtype=torch.float32),
            "routed_experts": _routes(msg_len, start),
        }
    ]


def _collate(batch):
    # Mirror the real pipeline order: the routing backfill (run inside
    # add_grpo_token_loss_masks_and_generation_logprobs) precedes the flatten.
    backfill_routed_experts(batch, sentinel=MISSING_ROUTE_SENTINEL)
    return batched_message_log_to_flat_message(
        batch,
        pad_value_dict={
            "token_ids": 0,
            "routed_experts": MISSING_ROUTE_SENTINEL,
        },
    )


def test_carry_collates_to_BTLK_with_sentinel_padding():
    # Two samples of different length -> the shorter is right-padded.
    flat, input_lengths = _collate(
        [_message_log(3, start=0), _message_log(5, start=1000)]
    )
    routed = flat["routed_experts"]
    assert routed.shape == (2, 5, _L, _K)
    assert routed.dtype == torch.int32
    # Valid regions preserved verbatim.
    assert torch.equal(routed[0, :3], _routes(3, 0))
    assert torch.equal(routed[1, :5], _routes(5, 1000))
    # Sample 0's right-pad (positions 3,4) is the missing-route sentinel.
    assert (routed[0, 3:5] == MISSING_ROUTE_SENTINEL).all()
    # Per-token column tracks the token length.
    assert input_lengths.tolist() == [3, 5]


def test_carry_absent_when_messages_have_no_routing():
    log = [
        {
            "role": "assistant",
            "token_ids": torch.arange(4, dtype=torch.int64),
            "generation_logprobs": torch.zeros(4, dtype=torch.float32),
        }
    ]
    flat, _ = _collate([log])
    assert "routed_experts" not in flat


def test_carried_column_binds_to_model():
    # The collated [B, T, L, K] column must satisfy the binder's layer-count
    # contract and bind onto each MoE block for a forward.
    flat, _ = _collate([_message_log(3, 0), _message_log(5, 1000)])
    routed = flat["routed_experts"]
    model = _moe_model(_L)
    with bind_router_replay(model, routed):
        for layer_idx, block in enumerate(model.blocks):  # type: ignore[attr-defined]
            assert block._replay_route_BLK is not None
            assert torch.equal(block._replay_route_BLK, routed[:, :, layer_idx, :])
    # Cleared on exit.
    for block in model.blocks:  # type: ignore[attr-defined]
        assert block._replay_route_BLK is None


def test_carried_column_layer_count_mismatch_raises():
    flat, _ = _collate([_message_log(3, 0)])
    routed = flat["routed_experts"]  # L == _L == 3
    model = _moe_model(_L - 1)       # model has fewer MoE blocks
    try:
        with bind_router_replay(model, routed):
            pass
    except ValueError as e:
        assert "does not match" in str(e)
    else:
        raise AssertionError("expected ValueError on layer-count mismatch")


def test_prompt_message_routing_stays_position_aligned():
    # Regression: a realistic log has a prompt message (tokens, no routing)
    # before the assistant turn. Without the routing backfill the assistant
    # routes are silently placed onto the prompt positions; with it they must
    # land at the assistant positions and the prompt stays sentinel.
    log = [
        {"role": "user", "token_ids": torch.arange(4, dtype=torch.int64)},
        {
            "role": "assistant",
            "token_ids": torch.arange(3, dtype=torch.int64),
            "routed_experts": _routes(3, 0),
        },
    ]
    flat, _ = _collate([log])
    assert flat["token_ids"].shape[1] == flat["routed_experts"].shape[1], (
        f"MISALIGNED length: token_ids={tuple(flat['token_ids'].shape)} "
        f"routed_experts={tuple(flat['routed_experts'].shape)}"
    )
    routed = flat["routed_experts"][0]
    # The assistant routes must land at the ASSISTANT token positions [4:7],
    # not at the prompt positions [0:4]; prompt positions must be sentinel.
    assert (routed[:4] == MISSING_ROUTE_SENTINEL).all(), (
        "prompt positions should be sentinel; assistant routing was misplaced "
        "onto the prompt region"
    )
    assert torch.equal(routed[4:7], _routes(3, 0)), (
        "assistant routing not aligned to the assistant token positions"
    )


def test_multi_turn_message_log_concatenates_routes():
    # Two assistant turns in one conversation -> routes concatenate along tokens,
    # mirroring how token_ids / generation_logprobs concatenate.
    log = _message_log(2, start=0) + _message_log(3, start=500)
    flat, input_lengths = _collate([log])
    routed = flat["routed_experts"]
    assert routed.shape == (1, 5, _L, _K)
    expected = torch.cat([_routes(2, 0), _routes(3, 500)], dim=0)
    assert torch.equal(routed[0], expected)
    assert input_lengths.tolist() == [5]
