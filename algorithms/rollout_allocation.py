"""Graded rollout allocation (Track F3a).

DAPO dynamic sampling is a binary gate: a prompt group whose reward variance is
zero contributes exactly zero gradient after baseline subtraction, so it is
dropped and the rest are kept. What it does *not* do is decide how to spend the
train batch when more rows survive the gate than the step can train on --
``dynamic_sampling`` takes the first ``train_prompts_size`` of them and discards
the surplus. That truncation is an ordering artifact: the rows it keeps are the
ones the dataloader happened to draw first, and the ones it throws away may carry
far more signal than the ones it kept.

This module spends the same budget by group variance instead. Every group that
survived the gate is guaranteed a floor of rows; the remainder is apportioned in
proportion to ``std ** variance_exponent``. Total rows trained on are unchanged,
so the step costs exactly what it cost before.

**Zero-variance groups stay dropped, and the floor does not resurrect them.**
Their advantage is identically zero for every row -- under both leave-one-out and
plain baselines, a group whose rewards are all equal has ``baseline == reward`` --
so budget spent on them buys no gradient at all. The floor applies to the
surviving set, where its job is to stop a proportional split from starving a
group with real but small variance.

**The allocation is across groups, never within one.** Selecting rows inside a
group by their own reward or advantage would be a non-random subsample
correlated with the quantity being estimated, and it would bias the group's
advantage. Within a group the rows kept are the first ``q`` in batch order, which
is exactly what head truncation already did -- the change is only *which groups*
the budget goes to. Choosing prompts by variance does bias prompt sampling, but
that is what dynamic sampling itself is: this grades a decision that is already
being made binarily.

**Grouping is by the same key the baseline used.** The group ids are assigned
from the same prompt tensor that ``calculate_baseline_and_std_per_prompt``
partitions on, via the same ``torch.unique(..., dim=0)`` call, so the two
partitions are identical by construction rather than by agreement.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from typing_extensions import NotRequired, TypedDict
from dockyard_rl.algorithms.config_access import read_config_value

# Column carrying each row's group id from the caller (which knows the prompt
# tensor) into the allocator (which does not). Popped off the batch before it
# reaches training.
GRADED_ALLOCATION_GROUP_KEY = "allocation_group"

# Trainer entry points that run dynamic sampling, which is the seam this
# extends. The async and data-plane paths refuse dynamic sampling outright
# (examples/run_grpo_swe.py), so this inherits their scope.
SUPPORTED_TRAINERS = frozenset({"sync"})

# Group ids are packed as `gen_batch_ordinal * _GROUP_ID_STRIDE + local_index`
# so that two generation batches accumulated into the same cache cannot merge
# groups that carry different baselines.
_GROUP_ID_STRIDE = 1 << 32


class GradedAllocationConfig(TypedDict):
    """Configuration for graded rollout allocation."""

    enabled: bool

    # Rows every surviving group is guaranteed before the remainder is
    # apportioned. 1 keeps every surviving prompt represented.
    min_rollouts_per_prompt: NotRequired[int]

    # Exponent on the group's std when apportioning the remainder. 0 is a
    # uniform split (the control); 1 is proportional; higher concentrates the
    # budget on the highest-variance groups.
    variance_exponent: NotRequired[float]


def _cfg(cfg: Any, key: str, default: Any) -> Any:
    """Read one key, tolerating a dict or a pydantic config (see config_access)."""
    value = read_config_value(cfg, key, default)
    return default if value is None else value


def graded_allocation_enabled(cfg: Optional[Any]) -> bool:
    """True when a config asks for graded allocation."""
    return bool(cfg) and bool(_cfg(cfg, "enabled", False))


def validate_graded_allocation_config(
    cfg: Optional[Any],
    *,
    use_dynamic_sampling: bool,
    trainer: str,
    context: str = "grpo.graded_allocation",
) -> None:
    """Refuse a configuration that cannot do what it says, before the run starts.

    Inert when disabled, so the default path is unchanged.

    Raises:
        ValueError: If the floor or exponent is out of range, dynamic sampling
            is off, or the trainer does not run the allocation seam.
    """
    if not graded_allocation_enabled(cfg):
        return

    floor = _cfg(cfg, "min_rollouts_per_prompt", 1)
    if not isinstance(floor, int) or isinstance(floor, bool) or floor < 0:
        raise ValueError(
            f"{context}.min_rollouts_per_prompt must be a non-negative integer, "
            f"got {floor!r}."
        )

    exponent = _cfg(cfg, "variance_exponent", 1.0)
    if exponent < 0.0:
        raise ValueError(
            f"{context}.variance_exponent must be >= 0, got {exponent}. A "
            "negative exponent sends the budget to the groups with the least "
            "variance, which is the opposite of the intent."
        )

    if not use_dynamic_sampling:
        raise ValueError(
            f"{context}.enabled=true requires grpo.use_dynamic_sampling=true. "
            "Graded allocation replaces the truncation dynamic sampling applies "
            "when more rollouts survive its variance gate than the train batch "
            "needs; with dynamic sampling off there is no surplus and no gate, "
            "so the setting would read as live and do nothing."
        )

    if trainer not in SUPPORTED_TRAINERS:
        raise ValueError(
            f"{context}.enabled=true is not supported on the {trainer!r} "
            "trainer. Dynamic sampling runs only on the sync trainer. "
            f"Supported: {sorted(SUPPORTED_TRAINERS)}."
        )


def assign_group_ids(
    prompt_ids: torch.Tensor,
    *,
    gen_batch_ordinal: int,
) -> torch.Tensor:
    """Per-row group ids, partitioned exactly as the baseline partitions.

    Args:
        prompt_ids:        (batch, seq) prompt tokens -- the same tensor passed
                           to ``calculate_baseline_and_std_per_prompt``.
        gen_batch_ordinal: Which generation batch these rows came from. Rows
                           accumulated into the dynamic-sampling cache across
                           several generation batches carry different baselines,
                           so identical prompts drawn twice must not merge.

    Returns:
        (batch,) int64 group ids.

    Raises:
        ValueError: If ``prompt_ids`` is not 2-D, or the batch holds more groups
            than the id packing can separate.
    """
    if prompt_ids.dim() != 2:
        raise ValueError(
            "assign_group_ids expects a 2-D (batch, seq) prompt tensor, got "
            f"shape {tuple(prompt_ids.shape)}. This is the same requirement "
            "calculate_baseline_and_std_per_prompt has -- it matches rows with "
            "`.all(1)` -- and the two must partition identically."
        )
    if gen_batch_ordinal < 0:
        raise ValueError(
            f"gen_batch_ordinal must be non-negative, got {gen_batch_ordinal}."
        )

    _, inverse = torch.unique(prompt_ids, dim=0, return_inverse=True)
    local = inverse.reshape(-1).to(torch.int64)
    if local.numel() and int(local.max().item()) >= _GROUP_ID_STRIDE:
        raise ValueError(
            f"Batch holds more than {_GROUP_ID_STRIDE} distinct prompts; group "
            "ids would collide across generation batches."
        )
    return local + int(gen_batch_ordinal) * _GROUP_ID_STRIDE


def _group_rows(group_ids: torch.Tensor) -> tuple[list[list[int]], list[int]]:
    """Rows per group, groups in first-appearance order."""
    order: list[int] = []
    rows: dict[int, list[int]] = {}
    for row, gid in enumerate(group_ids.reshape(-1).tolist()):
        gid = int(gid)
        if gid not in rows:
            rows[gid] = []
            order.append(gid)
        rows[gid].append(row)
    return [rows[gid] for gid in order], order


def _apportion(
    weights: list[float],
    headroom: list[int],
    remainder: int,
) -> list[int]:
    """Largest-remainder apportionment of `remainder` rows, capped by headroom.

    Deterministic: ties break on group index, and a group whose weight is zero
    is only reached by the round-robin fallback, which exists so an all-zero
    weight vector still spends the budget instead of silently returning short.
    """
    quotas = [0] * len(weights)
    active = [i for i in range(len(weights)) if headroom[i] > 0]

    while remainder > 0 and active:
        total_w = sum(weights[i] for i in active)
        if total_w <= 0.0:
            for i in active:
                if remainder == 0:
                    break
                quotas[i] += 1
                headroom[i] -= 1
                remainder -= 1
            active = [i for i in active if headroom[i] > 0]
            continue

        shares = [(i, remainder * weights[i] / total_w) for i in active]
        given = 0
        for i, share in shares:
            take = min(int(share), headroom[i])
            if take:
                quotas[i] += take
                headroom[i] -= take
                given += take
        remainder -= given

        if given == 0:
            # Every share is below one whole row; hand out single rows by
            # descending fractional share so the budget still lands.
            for i, _ in sorted(shares, key=lambda t: (-t[1], t[0])):
                if remainder == 0:
                    break
                if headroom[i] == 0:
                    continue
                quotas[i] += 1
                headroom[i] -= 1
                remainder -= 1

        active = [i for i in active if headroom[i] > 0]

    return quotas


def _diversity_factor(
    rows_per_group: list[list[int]],
    diversity: Optional[torch.Tensor],
    exponent: float,
) -> list[float]:
    """Per-group multiplier from routing diversity.

    Deliberately un-normalized, and the reason is a property of the consumers
    rather than of the signal: `_apportion` divides every weight by the active
    total, and the floor squeeze only ranks, so scaling the whole weight vector
    by any positive constant moves no row at all. That is what lets the signal
    stay in raw nats -- no normalizer is needed because none would be observable.
    An earlier revision divided by the batch mean and documented it as
    load-bearing; it was not, and the mutation sweep is what said so.

    Exponent 0 returns all ones (`v ** 0 == 1`), which is F3a's allocation
    unchanged; the explicit branch is a fast path, not a behavioural one.
    """
    if diversity is None or exponent == 0.0:
        return [1.0] * len(rows_per_group)

    flat = diversity.reshape(-1).to(torch.float64).tolist()
    per_group = [
        max(0.0, sum(flat[r] for r in rows) / len(rows)) for rows in rows_per_group
    ]
    if sum(per_group) <= 0.0:
        # No routing spread anywhere in the batch. This one IS observable:
        # every factor would be zero, the apportionment would see a zero total
        # and fall back to round-robin, and F3a's variance grading would be
        # silently discarded on precisely the batches the signal says nothing
        # about.
        return [1.0] * len(rows_per_group)
    return [value**exponent for value in per_group]


def allocate_rollout_budget(
    group_ids: torch.Tensor,
    weights: torch.Tensor,
    budget: int,
    *,
    min_rollouts_per_prompt: int = 1,
    variance_exponent: float = 1.0,
    diversity: Optional[torch.Tensor] = None,
    diversity_exponent: float = 0.0,
) -> list[int]:
    """Choose which `budget` rows to train on, graded by group variance.

    Args:
        group_ids:               (batch,) group id per row, from
                                 ``assign_group_ids``.
        weights:                 (batch,) per-row variance signal -- the ``std``
                                 column. Reduced to one weight per group by
                                 mean, since leave-one-out makes std vary within
                                 a group.
        budget:                  Rows to keep. When the batch has no more rows
                                 than this, every row is kept.
        min_rollouts_per_prompt: Rows guaranteed to each surviving group before
                                 the remainder is apportioned. When the floors
                                 alone exceed the budget, they are handed out to
                                 the highest-weight groups first.
        variance_exponent:       Exponent applied to each group's weight. 0
                                 gives a uniform split.
        diversity:               (batch,) optional routing-diversity signal
                                 (F3b). Never a reward -- it only moves budget
                                 between groups, so the policy is not paid for
                                 it and cannot game it.
        diversity_exponent:      Exponent on each group's diversity relative to
                                 the batch mean. 0.0 ignores `diversity`
                                 entirely.

    Returns:
        Row indices to keep, in ascending order.
    """
    total_rows = int(group_ids.reshape(-1).numel())
    if budget >= total_rows:
        return list(range(total_rows))
    if budget <= 0:
        return []

    rows_per_group, _ = _group_rows(group_ids)
    flat_weights = weights.reshape(-1).to(torch.float64).tolist()
    group_weight = [
        max(0.0, sum(flat_weights[r] for r in rows) / len(rows))
        for rows in rows_per_group
    ]
    available = [len(rows) for rows in rows_per_group]

    # One graded weight, read by both branches below. Ranking the floor squeeze
    # by the same quantity the apportionment uses matters once F3b's diversity
    # factor is in play: `std ** e` alone is monotone in `std`, so the two
    # rankings agreed before, and with a second factor they no longer would.
    factor = _diversity_factor(rows_per_group, diversity, diversity_exponent)
    graded = [
        (w**variance_exponent) * factor[i] for i, w in enumerate(group_weight)
    ]

    floor = max(0, int(min_rollouts_per_prompt))
    quotas = [min(floor, avail) for avail in available]

    if sum(quotas) > budget:
        # Not every group can be represented. Spend the floors on the groups
        # worth the most rather than on whichever came first.
        ranked = sorted(
            range(len(quotas)),
            key=lambda i: (-graded[i], -group_weight[i], i),
        )
        quotas = [0] * len(quotas)
        left = budget
        for i in ranked:
            if left <= 0:
                break
            take = min(available[i], floor, left)
            quotas[i] = take
            left -= take
    else:
        headroom = [available[i] - quotas[i] for i in range(len(quotas))]
        extra = _apportion(graded, headroom, budget - sum(quotas))
        quotas = [quotas[i] + extra[i] for i in range(len(quotas))]

    kept: list[int] = []
    for rows, quota in zip(rows_per_group, quotas):
        kept.extend(rows[:quota])
    kept.sort()
    return kept


def graded_allocation_metrics(
    group_ids: torch.Tensor,
    kept_indices: list[int],
    budget: int,
) -> dict[str, float]:
    """Report what the allocation did, against what truncation would have done.

    ``graded_allocation_rows_reallocated`` is the count of kept rows that head
    truncation would have discarded. Zero means the allocation agreed with
    arrival order for this step -- not that it failed to run.
    """
    rows_per_group, _ = _group_rows(group_ids)
    kept_set = set(kept_indices)
    per_group = [
        sum(1 for r in rows if r in kept_set) for rows in rows_per_group
    ]
    represented = [q for q in per_group if q > 0]
    truncated = set(range(min(budget, int(group_ids.reshape(-1).numel()))))

    return {
        "graded_allocation_groups": float(len(rows_per_group)),
        "graded_allocation_groups_represented": float(len(represented)),
        "graded_allocation_rows_per_group_min": float(min(represented, default=0)),
        "graded_allocation_rows_per_group_max": float(max(per_group, default=0)),
        "graded_allocation_rows_reallocated": float(
            len(kept_set - truncated)
        ),
    }
