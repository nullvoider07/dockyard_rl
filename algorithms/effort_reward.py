"""Difficulty-gated effort reward (Track F2).

The objective: spend fewer tokens where the group already solves the prompt, and
stop taxing long reasoning where it does not. Difficulty is read from the group's
own pass rate, so no difficulty labels exist anywhere.

    R_i      =  c_i · (1 − λ_eff · k_i)
    λ_eff    =  λ · max(p̄, floor)^γ

where ``c_i`` is correctness, ``k_i`` is normalized effort cost, and ``p̄`` is the
group's mean correctness.

**Why the penalty is multiplicative and not a subtracted tax.** A flat
``R = c − λ·k`` inverts on hard prompts: failure is cheap (a model that gives up
emits few tokens), success on a hard problem is expensive, and once
``λ·(k_success − k_failure) > 1`` the group-relative advantage ranks the rollouts
that gave up above the ones that solved it. The multiplicative form pins every
failure at exactly 0 and every success strictly above it, so cost only ever ranks
*within* the successful set.

**What makes that safe is the bound, not the difficulty scaling.** Success
outranks failure exactly when ``λ·(k_success − k_failure) < 1``, and since
``k ∈ [0, 1]`` by construction here, ``λ < 1`` is sufficient on its own. The
difficulty scaling is what makes the reward *useful*; the clamp and the bound are
what make it *safe*. Both are validated, and the clamp is the load-bearing one —
a cost that escapes its normalizer satisfies the bound on paper and breaks it in
fact.

**Prior art.** The two load-bearing choices — penalize only correct responses,
scale the penalty by group difficulty — are the validated core of ALP
(arXiv 2506.05256), which reports ~50% fewer tokens with 5.35× more spent on hard
problems than easy ones. ALP scales by ``max(acc, 1/G)``: linear, with a floor.
``difficulty_exponent`` and ``difficulty_floor`` are exposed so that setting is
reachable (γ=1, floor=1/G) rather than being a rival formulation, and so the
comparison is one run instead of an argument.

**Known gap.** CEEH (arXiv 2602.22642) finds that aggressive length optimization
collapses policy entropy and pushes the model toward confident shortcut
solutions, and pairs compression with entropy regularization on low-accuracy
prompts. No regularizer is implemented here. The stratified entropy metric this
module emits is the monitor, not the cure.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from typing_extensions import NotRequired, TypedDict

from dockyard_rl.algorithms.utils import calculate_baseline_and_std_per_prompt

# Estimators whose advantage is a per-prompt group baseline over `rewards`.
# Everything else either ignores `rewards` outright or has no group at all, so a
# shaped reward would be computed and then discarded -- a config that reads as
# live and does nothing.
SUPPORTED_ESTIMATORS = frozenset({"grpo"})

# Trainer entry points that run the reward pipeline this term belongs to.
SUPPORTED_TRAINERS = frozenset({"sync"})

CORRECTNESS_KEY = "unshaped_total_reward"


class EffortRewardConfig(TypedDict):
    """Configuration for the difficulty-gated effort reward."""

    enabled: bool

    # λ — overall penalty strength. Must be < 1: with cost clamped to [0, 1]
    # that is exactly the condition keeping every success above every failure.
    coefficient: NotRequired[float]

    # γ — how sharply the penalty scales with the group's pass rate. 1.0 is
    # ALP's linear scaling; 2.0 is markedly more permissive on hard prompts.
    difficulty_exponent: NotRequired[float]

    # ALP's `max(acc, 1/G)` floor, keeping a minimum penalty on groups that
    # scored zero. 0.0 disables it.
    difficulty_floor: NotRequired[float]

    # Token count that normalizes the cost to 1.0. Costs above it clamp.
    max_response_tokens: NotRequired[int]


def _cfg(cfg: Any, key: str, default: Any) -> Any:
    value = cfg.get(key, default) if hasattr(cfg, "get") else default
    return default if value is None else value


def validate_effort_reward_config(
    cfg: Optional[Any],
    *,
    estimator_name: str,
    trainer: str,
    context: str = "grpo.effort_reward",
) -> None:
    """Refuse a configuration that cannot do what it says, before the run starts.

    Inert when disabled, so the default path is unchanged.

    Raises:
        ValueError: If the coefficient is outside ``[0, 1)``, the normalizer is
            not positive, the exponent or floor is out of range, or the selected
            advantage estimator / trainer cannot consume the shaped reward.
    """
    if not cfg or not cfg.get("enabled", False):
        return

    coefficient = _cfg(cfg, "coefficient", 0.0)
    if not 0.0 <= coefficient < 1.0:
        raise ValueError(
            f"{context}.coefficient must be in [0, 1), got {coefficient}. "
            "Effort cost is clamped to [0, 1], so coefficient < 1 is exactly the "
            "bound that keeps every successful rollout ranked above every failed "
            "one; at or above 1 the group-relative advantage can invert on hard "
            "prompts and the policy learns to give up early."
        )

    max_response_tokens = _cfg(cfg, "max_response_tokens", 0)
    if max_response_tokens <= 0:
        raise ValueError(
            f"{context}.max_response_tokens must be a positive token count, got "
            f"{max_response_tokens}. It is the denominator that normalizes effort "
            "cost into [0, 1]; without it the coefficient bound is not a bound."
        )

    exponent = _cfg(cfg, "difficulty_exponent", 1.0)
    if exponent < 0.0:
        raise ValueError(
            f"{context}.difficulty_exponent must be >= 0, got {exponent}. "
            "A negative exponent inverts the difficulty scaling and taxes hard "
            "prompts hardest."
        )

    floor = _cfg(cfg, "difficulty_floor", 0.0)
    if not 0.0 <= floor <= 1.0:
        raise ValueError(
            f"{context}.difficulty_floor must be in [0, 1], got {floor}."
        )

    if estimator_name not in SUPPORTED_ESTIMATORS:
        raise ValueError(
            f"{context}.enabled=true requires an advantage estimator with a "
            f"per-prompt group baseline over rewards; grpo.adv_estimator.name is "
            f"{estimator_name!r}. gdpo reads only its reward/<name> components and "
            "ignores total_reward; opd derives its advantage from teacher "
            "logprobs and uses neither rewards nor prompt_ids; reinforce_plus_plus "
            "discards the group std and normalizes globally. Under any of those "
            "the shaped reward is computed and discarded. Supported: "
            f"{sorted(SUPPORTED_ESTIMATORS)}."
        )

    if trainer not in SUPPORTED_TRAINERS:
        raise ValueError(
            f"{context}.enabled=true is not supported on the {trainer!r} trainer. "
            "The reward pipeline it belongs to -- reward_scaling, reward_shaping, "
            "dynamic_sampling -- runs only on the sync trainer; the async and "
            "data-plane paths apply the invalid-action penalty and nothing else, "
            "so the term would never be applied. Supported: "
            f"{sorted(SUPPORTED_TRAINERS)}."
        )


def response_token_counts(message_log: Any) -> torch.Tensor:
    """Generated tokens per sample: every assistant turn, not only the first.

    Read from the message log at the reward stage rather than threaded from the
    rollout, so no per-sample column has to be added to three producers that
    share no constructor. `apply_reward_shaping`'s DAPO branch reads response
    length the same way -- but only from the *first* assistant message, which
    undercounts every multi-turn rollout. Effort is the whole trajectory.
    """
    counts = []
    for sample_log in message_log:
        total = 0
        for message in sample_log:
            if message.get("role") != "assistant":
                continue
            token_ids = message.get("token_ids")
            if token_ids is not None:
                total += int(token_ids.shape[0])
        counts.append(total)
    return torch.tensor(counts, dtype=torch.float32)


def normalize_effort_cost(
    token_counts: torch.Tensor, max_response_tokens: int
) -> torch.Tensor:
    """Map token counts into ``[0, 1]``, clamping rather than trusting the config.

    The clamp is the safety property, not a convenience: the coefficient bound
    assumes ``cost <= 1``, so a rollout longer than the configured normalizer
    must saturate instead of pushing the product past 1.
    """
    if max_response_tokens <= 0:
        raise ValueError(
            f"max_response_tokens must be positive, got {max_response_tokens}"
        )
    return (token_counts.float() / float(max_response_tokens)).clamp(0.0, 1.0)


def group_pass_rate(
    prompt_ids: torch.Tensor, correctness: torch.Tensor
) -> torch.Tensor:
    """Mean correctness of each sample's prompt group, broadcast per sample.

    This is a second call to the GRPO baseline helper, on the correctness column
    rather than the reward column. They are not the same quantity: the existing
    call returns the mean of `rewards`, and this term replaces the reward with
    `c·(1 − λ_eff·k)`, so the two coincide only before any shaping has happened.
    """
    baseline, _std = calculate_baseline_and_std_per_prompt(
        prompt_ids,
        correctness,
        torch.ones_like(correctness),
        leave_one_out_baseline=False,
    )
    return baseline


def effective_coefficient(
    pass_rate: torch.Tensor, cfg: Any
) -> torch.Tensor:
    """``λ_eff = λ · max(p̄, floor)^γ`` — full strength on easy, ~off on hard."""
    coefficient = float(_cfg(cfg, "coefficient", 0.0))
    exponent = float(_cfg(cfg, "difficulty_exponent", 1.0))
    floor = float(_cfg(cfg, "difficulty_floor", 0.0))
    gated = pass_rate.clamp(min=floor, max=1.0)
    return coefficient * gated.pow(exponent)


def apply_effort_reward(
    batch: Any,
    cfg: Optional[Any],
    *,
    prompt_ids: torch.Tensor,
) -> Any:
    """Apply the difficulty-gated effort term to ``batch["total_reward"]``.

    Mutates the batch in place and returns it, matching `apply_reward_shaping`
    and `apply_invalid_action_penalty`.

    Guarantees its own correctness column rather than inheriting one:
    `unshaped_total_reward` is written by `apply_reward_shaping` only *past* its
    `enabled` early return, so on a run with shaping off it does not exist. Both
    `p̄` and the multiplicand come from the pre-term reward, and the snapshot is
    left behind so dynamic sampling's variance gate keeps filtering on the task
    metric rather than on this term's spread.

    Raises:
        ValueError: If correctness falls outside ``[0, 1]``. The bound assumes
            it: a multi-reward environment sums its components unweighted, so
            `MathMultiRewardEnvironment` alone reaches 3.0.
    """
    if not cfg or not cfg.get("enabled", False):
        return batch

    rewards = batch["total_reward"]

    if CORRECTNESS_KEY not in batch:
        batch[CORRECTNESS_KEY] = rewards.clone()
    correctness = batch[CORRECTNESS_KEY]

    low = float(correctness.min())
    high = float(correctness.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(
            f"effort_reward requires correctness in [0, 1]; observed "
            f"[{low:.4g}, {high:.4g}]. The coefficient bound that keeps every "
            "success ranked above every failure assumes that range. A "
            "multi-reward environment sums its components unweighted (see "
            "MathMultiRewardEnvironment, which reaches 3.0), so scale the "
            "environment reward or leave this term disabled."
        )

    cost = normalize_effort_cost(
        response_token_counts(batch["message_log"]),
        int(_cfg(cfg, "max_response_tokens", 0)),
    ).to(rewards.device)

    pass_rate = group_pass_rate(prompt_ids, correctness)
    lam_eff = effective_coefficient(pass_rate, cfg).to(rewards.device)

    batch["effort_cost"] = cost
    # Published so telemetry needs no prompt-id tensor of its own. The trainer's
    # `input_ids` is deleted long before metrics are assembled, and reaching for
    # it there raises UnboundLocalError on every step, enabled or not.
    batch["effort_pass_rate"] = pass_rate.to(rewards.device)
    batch["total_reward"] = correctness * (1.0 - lam_eff * cost)
    return batch


def effort_reward_metrics(batch: Any) -> dict[str, float]:
    """Telemetry, including the entropy-collapse monitor CEEH argues for.

    Effort is reported stratified by difficulty because the mean hides the
    failure this term can cause: a run that shortens hard rollouts while
    lengthening easy ones has an unchanged mean and a broken hard tail.
    """
    if "effort_cost" not in batch or "effort_pass_rate" not in batch:
        return {}

    cost = batch["effort_cost"]
    pass_rate = batch["effort_pass_rate"]
    metrics = {
        "effort/cost_mean": float(cost.mean()),
        "effort/cost_max": float(cost.max()),
        "effort/pass_rate_mean": float(pass_rate.mean()),
    }
    for name, lo, hi in (("hard", 0.0, 0.34), ("medium", 0.34, 0.67), ("easy", 0.67, 1.01)):
        bucket = (pass_rate >= lo) & (pass_rate < hi)
        if bool(bucket.any()):
            metrics[f"effort/cost_mean_{name}"] = float(cost[bucket].mean())
    return metrics
