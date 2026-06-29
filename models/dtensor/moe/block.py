"""MoE block: router + grouped routed experts (+ optional shared experts).

Composes the gating network (``router.py``) with the grouped-GEMM experts
(``experts.py``). The block forward is:
  1. router -> per-token top-K expert ids + gating weights + full scores;
  2. build a routing map -> per-expert token counts (drives the dispatcher);
  3. routed experts (dispatch -> grouped compute -> combine);
  4. optional shared (always-on) expert, summed with the routed output.

The routing-map / token-count step (``routing_map_and_counts``) is pure and
CPU-tested. The full ``forward`` invokes ``GroupedExperts`` whose grouped-GEMM is
CUDA-only (HV-8), so end-to-end forward is device-bound.

Aux-loss-free load balancing (https://arxiv.org/abs/2408.15664): when
``load_balance_coeff`` is set, an ``expert_bias_E`` buffer biases the routing
CHOICE (not the gating weights). The buffer UPDATE is an optimizer pre-hook
applied outside the model and is a separate piece (not built here); without it
the bias stays zero (inert), so the default is off.

Distilled from torchtitan's ``MoE`` (pytorch/torchtitan, BSD-3-Clause;
Copyright (c) Meta Platforms, Inc. and affiliates).
"""

from __future__ import annotations

from typing import Optional, cast

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from dockyard_rl.models.dtensor.moe.experts import GroupedExperts
from dockyard_rl.models.dtensor.moe.router import TokenChoiceTopKRouter


def _routing_map(
    scores_BLE: torch.Tensor, topk_expert_ids_BLK: torch.Tensor
) -> torch.Tensor:
    """One-hot ``(B, L, E)`` bool map of which experts each token is routed to."""
    return torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
        -1, topk_expert_ids_BLK, True
    )


def routing_map_and_counts(
    scores_BLE: torch.Tensor, topk_expert_ids_BLK: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Routing map + per-expert token counts.

    ``scatter_`` cannot mix local Tensor / DTensor args, so when the router
    emits DTensors the scatter runs on locals via ``local_map`` and the result
    is re-wrapped. Returns ``(routing_map_BLE, num_tokens_per_expert_E)`` where
    the counts are summed over the batch+sequence dims.
    """
    if isinstance(topk_expert_ids_BLK, DTensor):
        assert isinstance(scores_BLE, DTensor), (
            "scores_BLE and topk_expert_ids_BLK must both be DTensors"
        )
        mapped = local_map(
            _routing_map,
            in_placements=(scores_BLE.placements, topk_expert_ids_BLK.placements),
            out_placements=(scores_BLE.placements,),
            device_mesh=scores_BLE.device_mesh,
        )
        routing_map_BLE = mapped(scores_BLE, topk_expert_ids_BLK)  # type: ignore[call-arg]
    else:
        routing_map_BLE = _routing_map(scores_BLE, topk_expert_ids_BLK)

    num_tokens_per_expert_E = routing_map_BLE.sum(dim=(0, 1))
    return routing_map_BLE, num_tokens_per_expert_E


class MoEBlock(nn.Module):
    """Router + grouped routed experts (+ optional shared experts).

    Args:
        router: The gating network.
        experts: Grouped routed experts.
        shared_experts: Optional always-on expert (dense FFN) summed with the
            routed output.
        num_experts: Total expert count (E) — for the load-balance buffers.
        load_balance_coeff: If set (> 0), enables the aux-loss-free
            ``expert_bias_E`` routing bias buffer. Its updater is external.
    """

    def __init__(
        self,
        router: TokenChoiceTopKRouter,
        experts: GroupedExperts,
        *,
        num_experts: int,
        shared_experts: Optional[nn.Module] = None,
        load_balance_coeff: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.router = router
        self.experts = experts
        self.shared_experts = shared_experts

        self.load_balance_coeff = load_balance_coeff
        if load_balance_coeff is not None:
            if load_balance_coeff <= 0.0:
                raise ValueError(
                    f"load_balance_coeff must be > 0, got {load_balance_coeff}"
                )
            self.register_buffer(
                "expert_bias_E",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias_E = None
        # Usage counter (non-persistent); accumulated each forward, also the
        # signal the external bias-update hook reads.
        self.register_buffer(
            "tokens_per_expert_E",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )
        # Transient MoE router-replay slot. A plain attribute (not a buffer, so
        # it never enters state_dict or moves with .to()): the router-replay
        # controller binds this layer's recorded route ``(B, L, K)`` before a
        # replayed forward and clears it after (set-and-consume). None on the
        # default path, so the forward and gating math are byte-unchanged.
        self._replay_route_BLK: Optional[torch.Tensor] = None

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        topk_scores_BLK, topk_expert_ids_BLK, scores_BLE = self.router(
            x_BLD, self.expert_bias_E, replay_route_BLK=self._replay_route_BLK
        )
        _, num_tokens_per_expert_E = routing_map_and_counts(
            scores_BLE, topk_expert_ids_BLK
        )

        @torch.no_grad()  # type: ignore[arg-type]
        def _update_token_counts() -> None:
            assert isinstance(self.tokens_per_expert_E, torch.Tensor)
            self.tokens_per_expert_E.add_(num_tokens_per_expert_E)

        _update_token_counts()

        out_BLD = self.experts(
            x_BLD, topk_scores_BLK, topk_expert_ids_BLK, num_tokens_per_expert_E
        )

        if self.shared_experts is not None:
            # nn.Module.__getattr__ is stubbed as returning Tensor | Module;
            # cast so the submodule call type-checks.
            shared = cast(nn.Module, self.shared_experts)
            out_BLD = out_BLD + shared(x_BLD)
        return out_BLD
