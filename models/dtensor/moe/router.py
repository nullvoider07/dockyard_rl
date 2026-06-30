"""Token-choice top-K router for MoE (the gating network).

Each token is scored against all experts (a small ``Linear`` gate), and routed to
its top-K experts. Optionally supports node-limited (group-limited) routing —
experts are partitioned into groups and only the top ``num_limited_groups`` groups
are considered before the top-K selection, which cuts cross-node traffic for
DeepSeek-style large-expert-count models.

Pure tensor logic (no grouped-GEMM, no all-to-all), so it runs and is unit-tested
on CPU. Distilled from torchtitan's ``TokenChoiceTopKRouter`` (pytorch/torchtitan,
BSD-3-Clause; Copyright (c) Meta Platforms, Inc. and affiliates), re-expressed as
a plain ``nn.Module`` without torchtitan's Config/Module protocol.

Shape suffixes: B=batch, L=seq, D=model dim, E=num experts, K=top-k.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn

ScoreFunc = Literal["softmax", "sigmoid"]

# Router-replay sentinel: a recorded route position the generation backend did
# not return (see models/generation/vllm/router_capture.py). The replay path
# falls back to the model's own routing for sentinel positions rather than
# indexing an out-of-range expert id. Canonical consumer-side definition; the
# producer side mirrors it (drift-guarded in tests).
MISSING_ROUTE_SENTINEL = -1


class TokenChoiceTopKRouter(nn.Module):
    """Top-K token-choice gating.

    Args:
        dim: Model dimension (D).
        num_experts: Total expert count (E).
        top_k: Experts per token (K).
        score_func: Gate activation — ``"sigmoid"`` (independent per-expert) or
            ``"softmax"`` (normalized over experts).
        route_norm: If True, renormalize the selected top-K gating scores to sum
            to 1 per token (after selection).
        route_scale: Multiplier applied to the final gating scores.
        num_expert_groups: If set, partition experts into this many contiguous
            groups for node-limited routing (must divide ``num_experts``).
        num_limited_groups: Number of groups kept per token when
            ``num_expert_groups`` is set; experts in other groups are masked out
            of the top-K choice.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        *,
        top_k: int = 1,
        score_func: ScoreFunc = "sigmoid",
        route_norm: bool = False,
        route_scale: float = 1.0,
        num_expert_groups: Optional[int] = None,
        num_limited_groups: Optional[int] = None,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError(
                f"top_k({top_k}) must be in [1, num_experts({num_experts})]"
            )
        if num_expert_groups is not None and num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.num_expert_groups = num_expert_groups
        self.num_limited_groups = num_limited_groups

    def _node_limited_scores(self, scores_BLE: torch.Tensor) -> torch.Tensor:
        """Mask experts outside the top ``num_limited_groups`` groups to -inf.

        Group score = sum of the group's top-2 expert scores (torchtitan
        convention). Tokens then choose top-K only among the kept groups.
        """
        assert self.num_expert_groups is not None
        assert self.num_limited_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts({self.num_experts}) must be divisible by "
                f"num_expert_groups({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(
                f"experts_per_group({experts_per_group}) must be >= 2"
            )
        grouped = scores_BLE.unflatten(
            -1, (self.num_expert_groups, experts_per_group)
        )
        top2, _ = grouped.topk(2, dim=-1)
        group_scores = top2.sum(dim=-1)
        _, keep_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        # Mask is True for groups to DROP; scatter False onto the kept groups.
        drop_mask = torch.ones_like(group_scores, dtype=torch.bool)
        drop_mask.scatter_(-1, keep_idx, False)
        return grouped.masked_fill(drop_mask.unsqueeze(-1), float("-inf")).flatten(-2)

    def _choose_topk_ids(
        self, scores_BLE: torch.Tensor, expert_bias_E: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Top-K expert ids from the (optionally biased, group-limited) scores."""
        scores_for_choice_BLE = (
            scores_BLE if expert_bias_E is None else scores_BLE + expert_bias_E
        )
        if self.num_expert_groups is not None:
            scores_for_choice_BLE = self._node_limited_scores(scores_for_choice_BLE)
        _, topk_expert_ids_BLK = torch.topk(
            scores_for_choice_BLE, k=self.top_k, dim=-1, sorted=False
        )
        return topk_expert_ids_BLK

    def _replayed_topk_ids(
        self,
        scores_BLE: torch.Tensor,
        expert_bias_E: Optional[torch.Tensor],
        replay_route_BLK: torch.Tensor,
    ) -> torch.Tensor:
        """Forced top-K ids from a recorded route, falling back on sentinels.

        ``replay_route_BLK`` is the generation-recorded selection ``(B, L, K)``.
        Positions marked ``MISSING_ROUTE_SENTINEL`` (a whole-K row the backend
        did not return) fall back to this model's own top-K choice rather than
        indexing an out-of-range expert; all other positions are forced to the
        recorded ids exactly.
        """
        if replay_route_BLK.shape[-1] != self.top_k:
            raise ValueError(
                f"replay_route_BLK last dim ({replay_route_BLK.shape[-1]}) must "
                f"equal top_k ({self.top_k})"
            )
        if replay_route_BLK.shape[:2] != scores_BLE.shape[:2]:
            raise ValueError(
                f"replay_route_BLK batch/seq {tuple(replay_route_BLK.shape[:2])} "
                f"must match scores {tuple(scores_BLE.shape[:2])}"
            )
        replay_ids_BLK = replay_route_BLK.to(torch.long)
        sentinel_BLK = replay_route_BLK == MISSING_ROUTE_SENTINEL
        if bool(sentinel_BLK.any()):
            computed_BLK = self._choose_topk_ids(scores_BLE, expert_bias_E)
            valid_BL1 = (~sentinel_BLK).all(dim=-1, keepdim=True)
            return torch.where(valid_BL1, replay_ids_BLK, computed_BLK)
        return replay_ids_BLK

    def forward(
        self,
        x_BLD: torch.Tensor,
        expert_bias_E: Optional[torch.Tensor] = None,
        replay_route_BLK: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to their top-K experts.

        Args:
            x_BLD: Input ``(B, L, D)``.
            expert_bias_E: Optional aux-loss-free load-balance bias ``(E,)``,
                added to the scores for the top-K CHOICE only (the returned
                gating scores still come from the unbiased scores).
            replay_route_BLK: Optional recorded routing ``(B, L, K)`` (MoE
                router-replay). When given, the top-K CHOICE is forced to these
                ids instead of computed from the scores (sentinel positions fall
                back to the computed choice); the gating WEIGHTS still come from
                this model's own ``scores_BLE``, so the gate gradient is intact.

        Returns:
            topk_scores_BLK: Gating weights for the selected experts ``(B, L, K)``.
            topk_expert_ids_BLK: Selected expert indices ``(B, L, K)`` (int64).
            scores_BLE: Full per-expert gating scores ``(B, L, E)``.
        """
        # Gate in float32 for load-balance stability (overrides an outer bf16
        # autocast). CPU autocast only supports bf16/fp16, so the float32 cast is
        # a no-op there (the gate params are already float32) — guard to CUDA to
        # apply the override on the training path without warning on CPU.
        if x_BLD.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                scores_BLE = self.gate(x_BLD)
        else:
            scores_BLE = self.gate(x_BLD)

        if self.score_func == "sigmoid":
            scores_BLE = torch.sigmoid(scores_BLE)
        elif self.score_func == "softmax":
            scores_BLE = F.softmax(scores_BLE, dim=-1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if replay_route_BLK is None:
            topk_expert_ids_BLK = self._choose_topk_ids(scores_BLE, expert_bias_E)
        else:
            topk_expert_ids_BLK = self._replayed_topk_ids(
                scores_BLE, expert_bias_E, replay_route_BLK
            )

        # Gating weights derive from the UNBIASED scores (bias is routing-only).
        topk_scores_BLK = scores_BLE.gather(dim=-1, index=topk_expert_ids_BLK)

        if self.route_norm:
            denom = topk_scores_BLK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_BLK = topk_scores_BLK / denom
        topk_scores_BLK = topk_scores_BLK * self.route_scale

        return topk_scores_BLK, topk_expert_ids_BLK, scores_BLE
