"""Routing diversity as an allocation signal (Track F3b).

Under a fixed ``top_k`` the model cannot spend more compute on a token: expert
activations are ``generated_tokens x num_layers x top_k`` exactly, a constant
multiple of the token count. What still varies is *which* experts fire across a
trajectory, and routing to more distinct experts engages more of the network's
parameters at identical FLOPs. That is the quantity this module measures.

**It is measured and never rewarded, and that is the design, not a limitation.**
Paying a policy for routing diversity is reward-hacking bait of the purest kind:
random routing maximizes it and destroys the specialization that makes it worth
having. Its value is as an *allocation* input the model cannot game precisely
because the reward never sees it -- nothing in the gradient rewards moving it, so
whatever it reports is incidental to what the policy is optimizing. A test in
``tests/unit/test_routing_diversity.py`` fails if the column ever reaches a
reward term.

**The statistic is per-layer routing entropy, not a distinct-expert count.** A
count of distinct experts is confounded by length: a longer rollout necessarily
touches more experts until it saturates at the expert count, so the "signal"
would largely re-rank rollouts by how long they are -- and length is exactly what
F2 already prices. The Shannon entropy of a layer's expert-usage histogram
converges to the trajectory's true routing entropy once picks exceed the expert
count (``tokens x top_k`` is in the thousands against an expert count in the
hundreds), so it is length-robust in the regime that matters. Short rollouts,
where picks are fewer than experts, are entropy-limited by their own length;
``routing_diversity_min_positions`` reports rather than hides them.

**Left in nats, deliberately un-normalized.** Normalizing to ``[0, 1]`` needs the
routed-expert count, which is a model-config read the driver does not otherwise
do -- and it would buy nothing, because the only consumer compares groups against
the batch mean, where any constant denominator cancels.

**Where the signal is read from.** The per-token route rides the message log, not
the batch: ``experience/rollouts.py:264`` attaches ``routed_experts`` to
rollout-generated assistant messages only, and the sentinel backfill that pads
the other messages (``grpo.py:1858``) runs *after* the reward stage. So reading
it here sees exactly the generated turns and nothing else -- the presence of the
key is itself the marker, which is sharper than a role check.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from typing_extensions import NotRequired, TypedDict

from dockyard_rl.models.generation.vllm.router_capture import MISSING_ROUTE_SENTINEL

# Column carrying the per-rollout signal from the reward stage to the allocator.
ROUTING_DIVERSITY_KEY = "routing_diversity"

# Trainer entry points that run the allocation seam this feeds.
SUPPORTED_TRAINERS = frozenset({"sync"})


class RoutingDiversityConfig(TypedDict):
    """Configuration for the routing-diversity allocation signal."""

    enabled: bool

    # Exponent on each group's diversity relative to the batch mean, applied as
    # a multiplicative factor on the allocation weight. 0.0 is inert -- the
    # allocation is F3a's variance grading, unchanged.
    allocation_exponent: NotRequired[float]


def _cfg(cfg: Any, key: str, default: Any) -> Any:
    value = cfg.get(key, default) if hasattr(cfg, "get") else default
    return default if value is None else value


def routing_diversity_enabled(cfg: Optional[Any]) -> bool:
    """True when a config asks for the routing-diversity signal."""
    return bool(cfg) and bool(_cfg(cfg, "enabled", False))


def validate_routing_diversity_config(
    cfg: Optional[Any],
    *,
    router_replay_enabled: bool,
    graded_allocation_enabled: bool,
    trainer: str,
    context: str = "grpo.routing_diversity",
) -> None:
    """Refuse a configuration that cannot do what it says, before the run starts.

    Inert when disabled, so the default path is unchanged.

    Raises:
        ValueError: If the exponent is out of range, the capture that produces
            the signal is off, no consumer would read it, or the trainer does
            not run the allocation seam.
    """
    if not routing_diversity_enabled(cfg):
        return

    exponent = _cfg(cfg, "allocation_exponent", 0.0)
    if exponent < 0.0:
        raise ValueError(
            f"{context}.allocation_exponent must be >= 0, got {exponent}. A "
            "negative exponent sends the budget to the groups that engage the "
            "least capacity, which is the opposite of the intent."
        )

    if not router_replay_enabled:
        raise ValueError(
            f"{context}.enabled=true requires policy.router_replay.enabled=true. "
            "The per-token route this reduces is captured only by the router-"
            "replay path (vLLM enable_return_routed_experts); without it no "
            "message carries a routed_experts key and the signal is identically "
            "zero for every rollout. Enabling router replay also picks up its "
            "own refusals -- non-vLLM backends and the data plane cannot "
            "surface the column at all."
        )

    if not graded_allocation_enabled:
        raise ValueError(
            f"{context}.enabled=true requires grpo.graded_allocation.enabled="
            "true, which is its only consumer. Measured and read by nothing, "
            "the column would be computed every step and discarded while the "
            "config reads as live."
        )

    if trainer not in SUPPORTED_TRAINERS:
        raise ValueError(
            f"{context}.enabled=true is not supported on the {trainer!r} "
            "trainer. The allocation seam that reads it runs only on the sync "
            f"trainer. Supported: {sorted(SUPPORTED_TRAINERS)}."
        )


def _turn_routes(message: Any) -> Optional[torch.Tensor]:
    """The usable ``[T-1, L, K]`` routes of one rollout-generated assistant turn.

    The final position is dropped: routing is a next-token quantity, so the route
    at a turn's last position either predicted a token outside the turn or -- for
    the trajectory's final token -- is the identity default ``arange(top_k)`` that
    ``routed_experts_empty`` fills padding with, which is a fabricated route and
    would inflate the histogram.
    """
    routes = message.get("routed_experts")
    if routes is None or not hasattr(routes, "dim"):
        return None
    if routes.dim() != 3 or routes.shape[0] < 2:
        return None
    return routes[:-1]


def sample_routing_diversity(sample_log: Any) -> tuple[float, int]:
    """Mean per-layer routing entropy (nats) over one rollout's generated turns.

    Returns:
        ``(diversity, positions)`` -- the entropy, and how many routed positions
        it was computed over. Zero positions yields zero diversity; the count is
        reported so a batch that is entropy-limited by its own length is visible
        rather than silently flat.
    """
    turns = [r for r in (_turn_routes(m) for m in sample_log) if r is not None]
    if not turns:
        return 0.0, 0

    stacked = torch.cat(turns, dim=0)
    positions = int(stacked.shape[0])
    if positions == 0:
        return 0.0, 0

    total = 0.0
    counted = 0
    for layer in range(int(stacked.shape[1])):
        picks = stacked[:, layer, :].reshape(-1).to(torch.int64)
        # MISSING_ROUTE_SENTINEL marks routes the backend failed to return. It
        # is not an expert, so it must leave the histogram rather than become a
        # phantom one that inflates entropy.
        picks = picks[picks != MISSING_ROUTE_SENTINEL]
        if picks.numel() == 0:
            continue
        counts = torch.bincount(picks).to(torch.float64)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        total += float(-(probs * probs.log()).sum().item())
        counted += 1

    if counted == 0:
        return 0.0, positions
    return total / counted, positions


def routing_diversity_from_message_log(
    message_log: Any,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Per-rollout routing diversity, or ``None`` when nothing carries routing.

    ``None`` rather than a zero column: a dense-model or capture-off run has no
    routing to reduce, and a column of zeros would read as "every rollout is
    perfectly concentrated" instead of "this quantity does not exist here".

    Returns:
        ``(diversity, positions)`` -- ``[B]`` float32 and ``[B]`` int64.
    """
    values: list[float] = []
    counts: list[int] = []
    for sample_log in message_log:
        diversity, positions = sample_routing_diversity(sample_log)
        values.append(diversity)
        counts.append(positions)

    position_tensor = torch.tensor(counts, dtype=torch.int64)
    if not any(counts):
        return None, position_tensor
    return torch.tensor(values, dtype=torch.float32), position_tensor


def routing_diversity_metrics(
    diversity: Optional[torch.Tensor],
    positions: torch.Tensor,
) -> dict[str, float]:
    """Report the signal without pricing it."""
    if diversity is None or diversity.numel() == 0:
        return {"routing_diversity_present": 0.0}
    return {
        "routing_diversity_present": 1.0,
        "routing_diversity_mean": float(diversity.mean().item()),
        "routing_diversity_min": float(diversity.min().item()),
        "routing_diversity_max": float(diversity.max().item()),
        "routing_diversity_min_positions": float(positions.min().item()),
    }
