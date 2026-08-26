"""Routed-expert capture alignment for MoE router-replay.

During generation an MoE router selects top-K experts per token. vLLM surfaces
this selection on ``CompletionOutput.routed_experts`` (an ``np.ndarray`` of
shape ``[tokens, num_moe_layers, topk]``, int16) when the engine is built with
``enable_return_routed_experts=True``.

``RequestOutput.prompt_routed_experts`` — the prompt-position counterpart — does
**not** exist in vLLM 0.27.x; a tree-wide search of 0.27.1 finds no such
attribute. Prompt-side routing is therefore unavailable and every sequence is
aligned from generated-token routes alone. That is the correct scope for
router replay regardless (the prompt is not sampled, so its routes never
contributed to a sampling decision), but the reduction is real and is reported
once per process by :func:`routed_experts_from_vllm_output` rather than being
hidden behind a ``getattr`` default.

This module aligns that per-token routing onto the trainer's right-padded layout
``[padded_length, L, topk]``, carrying the backend's own integer width, so it
can ride the rollout data dict and be
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

import logging
from typing import Any, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Interior route positions vLLM failed to return are filled with this so the
# trainer-side replay can detect incompleteness instead of replaying a default.
MISSING_ROUTE_SENTINEL = -1

# One report per process that the installed vLLM exposes no prompt-side routing.
# This adapter runs once per completion, so an unguarded log would flood.
_PROMPT_ROUTING_ABSENCE_REPORTED = False

ArrayLike = Union[torch.Tensor, np.ndarray, list]


# Widest fallback when the source carries no usable integer dtype of its own.
ROUTED_EXPERTS_FALLBACK_DTYPE = torch.int32


def _route_dtype(x: ArrayLike) -> torch.dtype:
    """The dtype an aligned route should carry for this source.

    vLLM returns ``routed_experts`` as int16; widening every capture to int32
    doubled a column that is ``[padded_length, L, topk]`` per sample and rides
    the rollout data dict through the object store. Nothing needs the extra
    width: the only consumer, ``MoERouter._replayed_topk_ids``, casts to
    ``torch.long`` before indexing.

    The source dtype is carried through when it is a *signed* integer type --
    ``MISSING_ROUTE_SENTINEL`` is -1, so an unsigned dtype cannot represent a
    missing route and is widened instead. Lists and floating sources fall back
    to int32.
    """
    dtype = getattr(x, "dtype", None)
    if dtype is not None and not isinstance(dtype, torch.dtype):
        # numpy dtype -> torch dtype, via a zero-copy empty view.
        try:
            dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
        except TypeError:
            dtype = None
    if (
        isinstance(dtype, torch.dtype)
        and not dtype.is_floating_point
        and not dtype.is_complex
        and dtype != torch.bool
        and dtype.is_signed
    ):
        return dtype
    return ROUTED_EXPERTS_FALLBACK_DTYPE


def _as_route_tensor(x: ArrayLike, device: Optional[torch.device]) -> torch.Tensor:
    """Coerce a routed-experts array to a tensor, carrying its integer dtype."""
    return torch.as_tensor(x, dtype=_route_dtype(x), device=device)


def routed_experts_empty(
    padded_length: int,
    num_layers: int,
    top_k: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = ROUTED_EXPERTS_FALLBACK_DTYPE,
) -> torch.Tensor:
    """Identity-route ``[padded_length, num_layers, top_k]`` integer tensor.

    The all-``arange(top_k)`` default route, used for samples/backends that
    return no routing (replay then degenerates to the trainer's own routing for
    those samples, i.e. a no-op correction). ``dtype`` must match the routing
    the rest of the batch carries, or the stack below refuses.
    """
    default_route = torch.arange(top_k, dtype=dtype, device=device)
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
    """Align per-token routing to ``[padded_length, L, topk]``.

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
        ``[padded_length, L, topk]`` integer tensor carrying the source's own
        dtype (or ``None`` if no routing was provided), optionally paired with a
        stats dict.
    """
    routed_t = None if routed is None else _as_route_tensor(routed, device)
    prompt_t = (
        None if prompt_routed is None else _as_route_tensor(prompt_routed, device)
    )

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
    full = routed_experts_empty(
        padded_length, num_layers, top_k, device=device, dtype=routed_t.dtype
    )

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
                    padded_length,
                    num_layers,
                    top_k,
                    device=device,
                    dtype=reference.dtype,
                )
            )
        else:
            if tuple(t.shape) != (padded_length, num_layers, top_k):
                raise ValueError(
                    "inconsistent aligned routed_experts shapes in batch: "
                    f"{tuple(t.shape)} vs "
                    f"{(padded_length, num_layers, top_k)}"
                )
            if t.dtype != reference.dtype:
                # torch.stack type-promotes rather than refusing, which would
                # silently widen the column back and hide a backend returning
                # a different width for one sample.
                raise ValueError(
                    "inconsistent aligned routed_experts dtypes in batch: "
                    f"{t.dtype} vs {reference.dtype}"
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

    Reads ``completion_output.routed_experts`` and delegates to
    :func:`align_routed_expert_indices`. Keeping the core on raw arrays leaves
    the alignment math unit-testable without constructing vLLM objects.

    ``request_output.prompt_routed_experts`` is read when the installed vLLM
    provides it and reported once when it does not. vLLM 0.27.x has no such
    attribute, so on the version dockyard ships against this is always the
    absent case and ``prompt_routed`` is always ``None``; the report exists so
    that shows up in a log rather than being inferred from a silent default.
    """
    routed = getattr(completion_output, "routed_experts", None)

    global _PROMPT_ROUTING_ABSENCE_REPORTED
    if hasattr(request_output, "prompt_routed_experts"):
        prompt_routed = request_output.prompt_routed_experts
    else:
        prompt_routed = None
        if not _PROMPT_ROUTING_ABSENCE_REPORTED:
            _PROMPT_ROUTING_ABSENCE_REPORTED = True
            logger.info(
                "This vLLM build exposes no RequestOutput.prompt_routed_experts "
                "(absent in 0.27.x); router replay is aligned from "
                "generated-token routes only. Prompt positions carry the "
                "identity default route.",
            )
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
