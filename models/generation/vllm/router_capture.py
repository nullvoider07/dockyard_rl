"""Routed-expert capture alignment for MoE router-replay.

During generation an MoE router selects top-K experts per token. vLLM 0.21
surfaces this selection on ``CompletionOutput.routed_experts`` (an
``np.ndarray`` of shape ``[tokens, num_moe_layers, topk]``, int16) when the
engine is built with ``enable_return_routed_experts=True``, plus an optional
``RequestOutput.prompt_routed_experts`` for the prompt positions. This module
aligns that per-token routing onto the trainer's right-padded sequence layout
``[padded_length, L, topk]`` so it can ride the rollout data dict and be
replayed in the trainer's MoE forward, removing the router-nondeterminism
component of the train/generation logprob mismatch.

Routing is a NEXT-TOKEN quantity: the route recorded at sequence position ``i``
drove the prediction of token ``i+1``, so only the first ``valid_length - 1``
positions carry a meaningful route; the final token and the right-padding take
the identity default route ``arange(topk)``. Genuinely-missing interior routes
(rare — observed only with vLLM prefix-caching plus chunked prefill omitting a
few rows) are marked with ``MISSING_ROUTE_SENTINEL`` so the consumer can detect
and reject them rather than silently replay a wrong route.

Pure tensor logic (no device collectives, no engine objects in the core) — runs
and is unit-tested on CPU.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch

# Interior route positions vLLM failed to return are filled with this so the
# trainer-side replay can detect incompleteness instead of replaying a default.
MISSING_ROUTE_SENTINEL = -1

ArrayLike = Union[torch.Tensor, np.ndarray, list]


def _as_int32(x: ArrayLike, device: Optional[torch.device]) -> torch.Tensor:
    """Coerce a routed-experts array (int16 ndarray / tensor / list) to int32."""
    return torch.as_tensor(x, dtype=torch.int32, device=device)


def routed_experts_empty(
    padded_length: int,
    num_layers: int,
    top_k: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Identity-route ``[padded_length, num_layers, top_k]`` int32 tensor.

    The all-``arange(top_k)`` default route, used for samples/backends that
    return no routing (replay then degenerates to the trainer's own routing for
    those samples, i.e. a no-op correction).
    """
    default_route = torch.arange(top_k, dtype=torch.int32, device=device)
    return (
        default_route.view(1, 1, -1)
        .expand(padded_length, num_layers, top_k)
        .clone()
    )


def align_routed_expert_indices(
    routed: Optional[ArrayLike],
    prompt_routed: Optional[ArrayLike] = None,
    *,
    valid_length: int,
    padded_length: int,
    device: Optional[torch.device] = None,
    require_complete: bool = False,
    allow_missing_fallback: bool = True,
    return_stats: bool = False,
) -> Union[
    Optional[torch.Tensor],
    tuple[Optional[torch.Tensor], dict[str, int]],
]:
    """Align per-token routing to ``[padded_length, L, topk]`` int32.

    Args:
        routed: vLLM ``CompletionOutput.routed_experts`` — ``[gen_tokens, L,
            topk]`` (the routes for the generated tokens, and for the prompt
            tokens too when ``prompt_routed`` is not surfaced separately). May be
            ``None`` when the backend returned no routing.
        prompt_routed: Optional ``RequestOutput.prompt_routed_experts``
            ``[prompt_tokens, L, topk]``; concatenated before ``routed`` to form
            the full-sequence routing.
        valid_length: Unpadded sequence length (prompt + response). Routing is a
            next-token quantity, so ``valid_length - 1`` positions carry a route.
        padded_length: Right-padded sequence length the output aligns to (must
            match ``output_ids``' second dim).
        device: Target device (default CPU).
        require_complete: If True, enforce that the backend returned a route for
            every non-final token (subject to ``allow_missing_fallback``).
        allow_missing_fallback: If True, tolerate missing interior routes by
            filling them with ``MISSING_ROUTE_SENTINEL`` instead of raising.
        return_stats: If True, also return a route-accounting dict.

    Returns:
        ``[padded_length, L, topk]`` int32 tensor (or ``None`` if no routing was
        provided), optionally paired with a stats dict.
    """
    routed_t = None if routed is None else _as_int32(routed, device)
    prompt_t = None if prompt_routed is None else _as_int32(prompt_routed, device)

    if prompt_t is not None and routed_t is not None:
        routed_t = torch.cat((prompt_t, routed_t), dim=0)
    elif prompt_t is not None:
        routed_t = prompt_t

    expected_routes = min(max(valid_length - 1, 0), padded_length)
    stats = {
        "actual_routes": 0,
        "expected_routes": expected_routes,
        "missing_routes": 0,
        "surplus_routes": 0,
    }

    if routed_t is None:
        return (None, stats) if return_stats else None
    if routed_t.dim() != 3:
        raise ValueError(
            "routed_experts must have shape [tokens, num_moe_layers, topk], "
            f"got {tuple(routed_t.shape)}"
        )

    actual = int(routed_t.shape[0])
    stats["actual_routes"] = actual
    stats["missing_routes"] = max(expected_routes - actual, 0)
    stats["surplus_routes"] = max(actual - (expected_routes + 1), 0)

    if require_complete and stats["missing_routes"] > 0 and not allow_missing_fallback:
        raise ValueError(
            "generation backend returned incomplete routed_experts for router "
            f"replay: routes={actual}, expected_at_least={expected_routes}, "
            f"valid_length={valid_length}, padded_length={padded_length}. The "
            "backend did not return a route for every non-final token in the "
            "prompt+response sequence."
        )
    max_allowed = expected_routes + 1
    if require_complete and actual > max_allowed:
        raise ValueError(
            "generation backend returned too many routed_experts routes for "
            f"router replay: routes={actual}, expected={expected_routes}, "
            f"max_allowed={max_allowed}, valid_length={valid_length}, "
            f"padded_length={padded_length}. Router replay allows at most one "
            "surplus final-token route."
        )

    num_layers = int(routed_t.shape[1])
    top_k = int(routed_t.shape[2])
    full = routed_experts_empty(padded_length, num_layers, top_k, device=device)

    routes_to_copy = min(expected_routes, actual)
    if routes_to_copy > 0:
        full[:routes_to_copy] = routed_t[:routes_to_copy]
    if stats["missing_routes"] > 0:
        full[routes_to_copy:expected_routes] = MISSING_ROUTE_SENTINEL

    return (full, stats) if return_stats else full


def stack_routed_experts(
    per_sample: list[Optional[torch.Tensor]],
    *,
    padded_length: int,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Stack per-sample aligned routes into a batch ``[B, padded_length, L, K]``.

    Each entry is an aligned ``[padded_length, L, topk]`` tensor (from
    :func:`align_routed_expert_indices`) or ``None`` for a sample whose backend
    returned no routing. Samples that are ``None`` are filled with the identity
    default route (:func:`routed_experts_empty`) using the ``(L, topk)`` of the
    first sample that did carry routing, so the batch is rectangular. Returns
    ``None`` when no sample carried routing (the routed_experts column is then
    simply absent and replay degenerates to a no-op).

    Args:
        per_sample: One entry per batch row; aligned route tensor or ``None``.
        padded_length: Sequence length every entry is (or will be) aligned to.
        device: Target device for synthesized empties / the stacked result.
    """
    reference = next((t for t in per_sample if t is not None), None)
    if reference is None:
        return None
    if reference.dim() != 3:
        raise ValueError(
            "aligned routed_experts entries must be [padded_length, L, topk], "
            f"got {tuple(reference.shape)}"
        )
    num_layers = int(reference.shape[1])
    top_k = int(reference.shape[2])
    rows: list[torch.Tensor] = []
    for t in per_sample:
        if t is None:
            rows.append(
                routed_experts_empty(
                    padded_length, num_layers, top_k, device=device
                )
            )
        else:
            if tuple(t.shape) != (padded_length, num_layers, top_k):
                raise ValueError(
                    "inconsistent aligned routed_experts shapes in batch: "
                    f"{tuple(t.shape)} vs "
                    f"{(padded_length, num_layers, top_k)}"
                )
            rows.append(t.to(device) if device is not None else t)
    return torch.stack(rows, dim=0)


def routed_experts_from_vllm_output(
    request_output: Any,
    completion_output: Any,
    *,
    valid_length: int,
    padded_length: int,
    device: Optional[torch.device] = None,
    require_complete: bool = False,
    allow_missing_fallback: bool = True,
    return_stats: bool = False,
) -> Union[
    Optional[torch.Tensor],
    tuple[Optional[torch.Tensor], dict[str, int]],
]:
    """Adapter: align routing straight off vLLM's request/completion objects.

    Reads ``completion_output.routed_experts`` and
    ``request_output.prompt_routed_experts`` (both optional) and delegates to
    :func:`align_routed_expert_indices`. Keeping the core on raw arrays leaves
    the alignment math unit-testable without constructing vLLM objects.
    """
    routed = getattr(completion_output, "routed_experts", None)
    prompt_routed = getattr(request_output, "prompt_routed_experts", None)
    return align_routed_expert_indices(
        routed,
        prompt_routed,
        valid_length=valid_length,
        padded_length=padded_length,
        device=device,
        require_complete=require_complete,
        allow_missing_fallback=allow_missing_fallback,
        return_stats=return_stats,
    )
