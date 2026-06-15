"""Grouped-GEMM routed-expert compute for MoE (SwiGLU).

The routed-expert FFN runs as a single grouped matmul over the local experts via
``torch._grouped_mm`` instead of a per-expert Python loop. Distilled from
torchtitan's GroupedExperts (pytorch/torchtitan, BSD-3-Clause; Copyright (c)
Meta Platforms, Inc. and affiliates).

Expert weights are 3D ``(num_experts, *, *)`` and shard ``Shard(0)`` over the EP
mesh (see ``sharding.py``); under EP, ``num_experts`` here is the LOCAL expert
count. ``forward`` orchestrates dispatch -> grouped compute -> combine via an
injected token dispatcher.

NOTE: ``torch._grouped_mm`` is a CUDA-only kernel (private torch API) — the
expert compute cannot run/test on CPU. See hardware-deferred-validation.md
(HV-8). The token dispatch/combine (``dispatch.py``) is pure and CPU-tested.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from dockyard_rl.models.dtensor.moe.dispatch import LocalTokenDispatcher


class GroupedExperts(nn.Module):
    """SwiGLU routed experts computed as a grouped matmul.

    Args:
        dim: Model dimension (D).
        hidden_dim: FFN intermediate dimension (F).
        num_experts: Local expert count (E / ep under expert parallelism).
        dispatcher: Token dispatcher providing dispatch()/combine().
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        dispatcher: LocalTokenDispatcher,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.w1_EFD = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2_EDF = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3_EFD = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.dispatcher = dispatcher

    def _experts_forward(
        self, x_RD: torch.Tensor, num_tokens_per_expert_E: torch.Tensor
    ) -> torch.Tensor:
        """Grouped SwiGLU over expert-contiguous tokens (CUDA-only)."""
        if isinstance(self.w1_EFD, DTensor):
            # Dynamic EP shapes can't be expressed as DTensors; compute on locals.
            w1_EFD = self.w1_EFD.to_local()
            w2_EDF = self.w2_EDF.to_local()  # type: ignore[attr-defined]
            w3_EFD = self.w3_EFD.to_local()  # type: ignore[attr-defined]
        else:
            w1_EFD, w2_EDF, w3_EFD = self.w1_EFD, self.w2_EDF, self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)

        h_RF = F.silu(
            torch._grouped_mm(
                x_RD.bfloat16(), w1_EFD.bfloat16().transpose(-2, -1), offs=offsets_E
            )
        )
        h_RF = h_RF * torch._grouped_mm(
            x_RD.bfloat16(), w3_EFD.bfloat16().transpose(-2, -1), offs=offsets_E
        )
        return torch._grouped_mm(
            h_RF, w2_EDF.bfloat16().transpose(-2, -1), offs=offsets_E
        ).type_as(x_RD)

    def forward(
        self,
        x_BLD: torch.Tensor,
        topk_scores_BLK: torch.Tensor,
        topk_expert_ids_BLK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, grouped-compute, and combine back."""
        B, L, D = x_BLD.shape
        K = topk_scores_BLK.size(-1)
        T = B * L
        x_TD = x_BLD.view(T, D)
        topk_scores_TK = topk_scores_BLK.view(T, K)
        topk_expert_ids_TK = topk_expert_ids_BLK.view(T, K)

        routed_input_RD, counts_E, metadata = self.dispatcher.dispatch(
            x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
        )
        routed_output_RD = self._experts_forward(routed_input_RD, counts_E)
        out_TD = self.dispatcher.combine(routed_output_RD, metadata, x_TD)
        return out_TD.view(B, L, D)
