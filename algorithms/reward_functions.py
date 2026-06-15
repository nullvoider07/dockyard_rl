"""Reward shaping functions for Project Dockyard.

Currently supports:
  - DAPO-style overlong response penalty (linear decay over buffer zone)
  - Stop-properly penalty (scale reward of truncated responses)
  - Invalid-action penalty (per-sample subtraction from rollout-time
    invalid-action / malformed-thinking verdict counts, #2656)
"""

from typing import NotRequired, TypedDict
import torch
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.rewards.invalid_action import InvalidActionPenaltyConfig

class RewardShapingConfig(TypedDict):
    """Configuration for reward function processing.

    Enables custom reward shaping, currently supporting DAPO-style
    penalties for responses that exceed the maximum response length
    threshold.
    """

    enabled: bool

    # Length of the buffer zone for penalising responses that exceed
    # max_response_length.  Responses longer than
    # overlong_buffer_length + max_response_length receive the maximum penalty.
    overlong_buffer_length: NotRequired[int]

    # Maximum penalty applied to responses exceeding max_response_length.
    overlong_buffer_penalty: NotRequired[float]

    # Maximum response length threshold.  Responses exceeding this length
    # are penalised.
    max_response_length: NotRequired[int]

    # Stop-properly penalty: scale factor for rewards of truncated responses
    # (range 0-1).  When set to 0, truncated responses get zero reward.
    # When set to 1, no penalty is applied (default behaviour).
    stop_properly_penalty_coef: NotRequired[float | None]

def apply_reward_shaping(
    batch: BatchedDataDict,
    cfg:   RewardShapingConfig,
) -> BatchedDataDict:
    """Apply reward shaping penalties to the batch.

    Supports two mutually exclusive modes:

    **Stop-properly penalty** (``stop_properly_penalty_coef`` is set):
        Scales the reward of truncated responses by the given coefficient.
        Overlong buffer parameters are ignored and a warning is printed.

    **DAPO overlong penalty** (``stop_properly_penalty_coef`` is None):
        Linearly penalises responses that exceed
        ``max_response_length - overlong_buffer_length`` tokens.
        Based on https://arxiv.org/pdf/2503.14476.

    Args:
        batch: BatchedDataDict containing "total_reward" and,
               for DAPO mode, "message_log".
        cfg:   RewardShapingConfig.

    Returns:
        The same BatchedDataDict with "total_reward" updated in-place.
    """
    rewards = batch["total_reward"]

    if not cfg["enabled"]:
        return batch

    # Preserve the pre-shaping reward so dynamic sampling can compute std on the
    # raw task metric (see calculate_baseline_and_std_per_prompt std_rewards).
    batch["unshaped_total_reward"] = rewards.clone()

    # DAPO overlong penalty
    stop_coef = cfg.get("stop_properly_penalty_coef", None)
    if stop_coef is not None:
        assert 0 <= stop_coef <= 1, (
            f"stop_properly_penalty_coef must be in [0, 1], got {stop_coef}"
        )
        # Warn if DAPO overlong params are also set — they are ignored.
        ignored = [
            k for k in ("overlong_buffer_length", "overlong_buffer_penalty",
                        "max_response_length")
            if cfg.get(k) is not None
        ]
        if ignored:
            print(
                f"[WARN] stop_properly_penalty_coef is set, so the following DAPO "
                f"overlong parameters are ignored: {', '.join(ignored)}. "
                "Set stop_properly_penalty_coef=null to use DAPO overlong reward "
                "shaping instead.",
                flush=True,
            )

        truncated = batch.get("truncated")
        assert truncated is not None, (
            "truncated field not found in batch"
        )
        if isinstance(truncated, list):
            truncated = torch.tensor(truncated, dtype=torch.bool, device=rewards.device)
        else:
            truncated = truncated.to(device=rewards.device)

        num_truncated = truncated.sum().item()
        if num_truncated > 0:
            original = rewards.clone()
            rewards  = torch.where(truncated, rewards * stop_coef, rewards)
            batch["total_reward"] = rewards
            print(
                f"[INFO] stop properly penalty applied: "
                f"{num_truncated}/{len(truncated)} samples truncated, "
                f"coef={stop_coef}, "
                f"original_reward_mean={original[truncated].mean().item():.4f}, "
                f"shaped_reward_mean={rewards[truncated].mean().item():.4f}",
                flush=True,
            )
        else:
            print(
                "[INFO] stop properly penalty: no truncated samples "
                "(truncation_rate=0)",
                flush=True,
            )
        return batch

    # DAPO overlong penalty
    if any(
        cfg.get(k) is None
        for k in ("overlong_buffer_length", "overlong_buffer_penalty",
                  "max_response_length")
    ):
        raise ValueError(
            "Reward shaping is enabled but only DAPO reward shaping is "
            "currently supported.  Please ensure overlong_buffer_length, "
            "overlong_buffer_penalty, and max_response_length are properly "
            "configured."
        )

    overlong_buffer_length  = cfg.get("overlong_buffer_length")
    overlong_buffer_penalty = cfg.get("overlong_buffer_penalty")
    max_response_length     = cfg.get("max_response_length")
    assert overlong_buffer_length  is not None
    assert overlong_buffer_penalty is not None
    assert max_response_length     is not None
    assert overlong_buffer_penalty >= 0, (
        f"{overlong_buffer_penalty=} must be >= 0"
    )

    expected_response_length = max_response_length - overlong_buffer_length

    assert len(batch["message_log"]) == len(rewards), (
        "The number of messages in the batch must match the number of rewards"
    )

    updated_rewards = torch.zeros_like(rewards)
    for i, message_log in enumerate(batch["message_log"]):
        response_length = None
        for message in message_log:
            if message["role"] == "assistant":
                response_length = message["token_ids"].shape[0]
                break
        assert response_length is not None, (
            "Assistant response not found during reward shaping"
        )

        exceed_length    = response_length - expected_response_length
        overlong_penalty = min(
            -exceed_length / overlong_buffer_length * overlong_buffer_penalty, 0
        )
        updated_rewards[i] = rewards[i] + overlong_penalty

    batch["total_reward"] = updated_rewards
    return batch

def apply_invalid_action_penalty(
    batch: BatchedDataDict,
    cfg:   "InvalidActionPenaltyConfig | None",
) -> BatchedDataDict:
    """Subtract the per-sample invalid-action / malformed-thinking penalty.

    Operates on the per-sample verdict counts produced at rollout time
    (``invalid_action_count`` / ``malformed_thinking_count``); pure arithmetic,
    so it runs identically on the driver batch and on the data-plane
    ``driver_carry``. Disabled config is a strict no-op.

    Args:
        batch: BatchedDataDict containing "total_reward" and, when enabled,
               the two count fields.
        cfg:   InvalidActionPenaltyConfig or None.

    Returns:
        The same BatchedDataDict with "total_reward" updated in-place.
    """
    if cfg is None or not cfg.get("enabled", False):
        return batch

    missing = [
        k for k in ("invalid_action_count", "malformed_thinking_count")
        if k not in batch
    ]
    if missing:
        raise ValueError(
            f"invalid_action_penalty is enabled but the rollout did not produce "
            f"{missing}. The penalty config must be threaded to the rollout "
            f"(run_multi_turn_rollout / run_async_multi_turn_rollout via "
            f"invalid_action_cfg) so verdict counts are collected."
        )

    invalid_pen   = float(cfg.get("invalid_action_penalty", 0.0))
    malformed_pen = float(cfg.get("malformed_thinking_penalty", 0.0))
    assert invalid_pen >= 0 and malformed_pen >= 0, (
        f"penalties must be >= 0 (subtracted from reward), got "
        f"{invalid_pen=} {malformed_pen=}"
    )

    rewards = batch["total_reward"]

    # Preserve the pre-penalty reward for dynamic-sampling std on the raw task
    # metric, matching the apply_reward_shaping contract. Only set when no
    # earlier shaping stage already preserved it.
    if "unshaped_total_reward" not in batch:
        batch["unshaped_total_reward"] = rewards.clone()

    invalid_counts   = batch["invalid_action_count"].to(rewards.dtype)
    malformed_counts = batch["malformed_thinking_count"].to(rewards.dtype)
    penalized = rewards - invalid_pen * invalid_counts - malformed_pen * malformed_counts

    floor = cfg.get("reward_floor")
    if floor is not None:
        penalized = torch.maximum(
            penalized, torch.full_like(penalized, float(floor))
        )

    n_flagged = int(((invalid_counts > 0) | (malformed_counts > 0)).sum().item())
    if n_flagged > 0:
        print(
            f"[INFO] invalid action penalty applied: {n_flagged}/{len(penalized)} "
            f"samples flagged (invalid_action turns={int(invalid_counts.sum().item())}, "
            f"malformed_thinking turns={int(malformed_counts.sum().item())}), "
            f"reward_mean {rewards.mean().item():.4f} -> {penalized.mean().item():.4f}",
            flush=True,
        )

    batch["total_reward"] = penalized
    return batch