"""Advantage estimators for RL algorithms.

Provides:
  GRPOAdvantageEstimator         — leave-one-out baseline, per-prompt normalisation
  GDPOAdvantageEstimator         — multi-reward GRPO: a per-component leave-one-out
                                   baseline for each reward axis, summed then
                                   renormalised (for multi-objective environments)
  ReinforcePlusPlusAdvantageEstimator — optional baseline subtraction + KL in reward

References:
  ProRL v2: https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/
  Reinforce++: https://arxiv.org/abs/2501.03262
"""

import torch

from dockyard_rl.algorithms.loss.interfaces import LossType
from dockyard_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
from dockyard_rl.algorithms.loss.utils import calculate_kl
from dockyard_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    get_gdpo_reward_component_keys,
)

class GRPOAdvantageEstimator:
    """GRPO-style advantage estimator with leave-one-out baseline.

    Computes advantages over all responses for each prompt (same prompt,
    N generations).  Normalisation is per-prompt, not global.
    """

    def __init__(
        self,
        estimator_config: dict,
        loss_config:      ClippedPGLossConfig,
    ) -> None:
        self.use_leave_one_out_baseline = estimator_config["use_leave_one_out_baseline"]
        self.normalize_rewards          = estimator_config["normalize_rewards"]

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        **kwargs,
    ) -> torch.Tensor:
        """Compute GRPO advantages.

        Args:
            prompt_ids: (batch,)        Identifies which prompt each sample belongs to.
            rewards:    (batch,)        Per-sample scalar reward.
            mask:       (batch, seq)    1 = valid response token, 0 = padding.
                                        Used only for expanding to token-level shape.
            **kwargs:   Ignored.

        Returns:
            Advantages tensor of shape (batch, seq).
        """
        baseline, std = calculate_baseline_and_std_per_prompt(
            prompt_ids,
            rewards,
            torch.ones_like(rewards),
            leave_one_out_baseline=self.use_leave_one_out_baseline,
        )
        advantages = (rewards - baseline).unsqueeze(-1)

        if self.normalize_rewards:
            epsilon = 1e-6
            non_zero_std_mask = std > 0
            advantages[non_zero_std_mask] = advantages[non_zero_std_mask] / (
                std.unsqueeze(-1)[non_zero_std_mask] + epsilon
            )

        return advantages.expand(mask.shape)

class GDPOAdvantageEstimator:
    """Multi-reward (multi-objective) advantage estimator — a GRPO generalisation.

    Where GRPO scores each rollout by a single scalar reward, GDPO handles an
    environment that emits several reward components per rollout (reward1,
    reward2, …) — e.g. tests-pass AND lint-clean AND a runtime budget, or several
    independent graders. For each component it computes a GRPO-style per-prompt
    leave-one-out baseline (and optionally per-prompt std-normalises), sums the
    per-component advantages, then renormalises the total to zero mean / unit std.
    Keeping each objective on its own baseline before combining stops a component
    with a different scale or hit-rate from dominating.

    Selected by ``grpo.adv_estimator.name='gdpo'``. Requires at least two reward
    components in the batch; raises ValueError otherwise (use 'grpo' for a single
    reward). The "GDPO" name is not expanded in this codebase — read it as
    "multi-reward GRPO".
    """

    def __init__(
        self,
        estimator_config: dict,
        loss_config:      ClippedPGLossConfig,
    ) -> None:
        self.use_leave_one_out_baseline = estimator_config["use_leave_one_out_baseline"]
        self.normalize_rewards          = estimator_config["normalize_rewards"]

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        repeated_batch,
        **kwargs,
    ) -> torch.Tensor:
        """Compute GDPO advantages.

        Args:
            prompt_ids:     (batch,)       Per-sample prompt identifier.
            rewards:        Unused; present for interface consistency.
            mask:           (batch, seq)   Response token mask.
            repeated_batch: BatchedDataDict containing reward1, reward2, … keys.
            **kwargs:       Ignored.

        Returns:
            Advantages tensor of shape (batch, seq).
        """
        reward_component_keys = get_gdpo_reward_component_keys(repeated_batch)
        if len(reward_component_keys) < 2:
            raise ValueError(
                f"GDPO requires multiple reward components (reward1, reward2, …). "
                f"This batch has {len(reward_component_keys)} component(s). "
                "Switch to GRPO by setting grpo.adv_estimator.name='grpo'."
            )

        valid        = torch.ones_like(repeated_batch[reward_component_keys[0]])
        leave_one_out = self.use_leave_one_out_baseline

        assert prompt_ids.shape[0] == valid.shape[0], (
            f"prompt_ids must match reward batch size; "
            f"got {prompt_ids.shape[0]} vs {valid.shape[0]}"
        )

        advantage_parts = []
        for key in reward_component_keys:
            r = repeated_batch[key]
            base, std_k = calculate_baseline_and_std_per_prompt(
                prompt_ids, r, valid,
                leave_one_out_baseline=leave_one_out,
            )
            adv_k = (r - base).unsqueeze(-1)
            if self.normalize_rewards:
                epsilon = 1e-6
                nz_mask = std_k > 0
                adv_k[nz_mask] = adv_k[nz_mask] / (
                    std_k.unsqueeze(-1)[nz_mask] + epsilon
                )
            advantage_parts.append(adv_k)

        advantages = sum(advantage_parts)

        # Normalise combined advantage to zero mean and unit std.
        adv_std = advantages.std()
        if adv_std > 0:
            advantages = (advantages - advantages.mean()) / adv_std
        else:
            advantages = advantages - advantages.mean()

        return advantages.expand(mask.shape)

class ReinforcePlusPlusAdvantageEstimator:
    """Reinforce++ advantage estimator with optional baseline and KL in reward.

    Args:
        minus_baseline:    If True, subtract per-prompt mean from rewards.
        use_kl_in_reward:  If True, add KL penalty to advantages (token-level)
                           instead of the loss term.
    """

    def __init__(
        self,
        estimator_config: dict,
        loss_config:      ClippedPGLossConfig,
    ) -> None:
        self.minus_baseline    = estimator_config["minus_baseline"]
        self.use_kl_in_reward  = loss_config.use_kl_in_reward
        self.kl_coef           = loss_config.reference_policy_kl_penalty
        self.kl_type           = loss_config.reference_policy_kl_type

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        logprobs_policy=None,
        logprobs_reference=None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute Reinforce++ advantages.

        Args:
            prompt_ids:          (batch,)      Per-sample prompt identifier.
            rewards:             (batch,)      Scalar reward.
            mask:                (batch, seq)  Response token mask.
            logprobs_policy:     (batch, seq)  Policy log-probs (required if
                                              use_kl_in_reward).
            logprobs_reference:  (batch, seq)  Reference log-probs (required if
                                              use_kl_in_reward).
            **kwargs:            Ignored.

        Returns:
            Advantages tensor of shape (batch, seq), globally normalised over
            valid tokens.
        """
        if self.minus_baseline:
            mean, _ = calculate_baseline_and_std_per_prompt(
                prompt_ids,
                rewards,
                torch.ones_like(rewards),
                leave_one_out_baseline=False,
            )
            adv = rewards - mean
        else:
            adv = rewards

        adv = adv.unsqueeze(-1).expand(mask.shape)

        # Optional token-level KL penalty added to the advantage signal.
        if (
            self.use_kl_in_reward
            and logprobs_policy is not None
            and logprobs_reference is not None
        ):
            kl  = calculate_kl(logprobs_policy, logprobs_reference, kl_type=self.kl_type)
            adv = adv - self.kl_coef * kl

        # Global normalisation across the batch (valid tokens only).
        adv_mean  = (adv * mask).sum() / mask.sum()
        adv_var   = ((adv - adv_mean).pow(2) * mask).sum() / mask.sum()
        adv_rstd  = adv_var.clamp(min=1e-8).rsqrt()
        adv       = (adv - adv_mean) * adv_rstd

        return adv.expand(mask.shape)


class OPDAdvantageEstimator:
    """Multi-Teacher On-Policy Distillation (MOPD) advantage estimator (arXiv:2601.02780).

    Token-level distillation advantage (Eq. 8):

        Â_MOPD,t = sg[log π_teacher - log π_student]

    The importance-sampling truncation (the hard gate on the training-to-inference
    ratio) is handled separately by ICE-POP mode in ClippedPGLoss, not here. The
    loss should be configured with:

        disable_ppo_ratio: true                          (REINFORCE, no PPO ratio)
        use_importance_sampling_correction: true
        truncated_importance_sampling_type: icepop
        truncated_importance_sampling_ratio_min: <eps_low>
        truncated_importance_sampling_ratio: <eps_high>

    Selected by ``grpo.adv_estimator.name='opd'``. Requires per-token teacher
    logprobs (``teacher_logprobs``) and student training-engine logprobs
    (``logprobs_policy``, dockyard's ``prev_logprobs``).
    """

    def __init__(
        self,
        estimator_config: dict,
        loss_config: ClippedPGLossConfig,
    ) -> None:
        self.last_metrics: dict[str, float] = {}

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        logprobs_policy=None,
        teacher_logprobs=None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute OPD token-level distillation advantages.

        Args:
            prompt_ids: (batch,) prompt IDs (unused; interface compatibility).
            rewards:    (batch,) rewards (unused for pure distillation).
            mask:       (batch, seq) 1 = valid response token, 0 = padding.
            logprobs_policy:  (batch, seq) student training-engine logprobs
                              (dockyard's prev_logprobs). Required.
            teacher_logprobs: (batch, seq) teacher model logprobs. Required.
            **kwargs:   Ignored (e.g. logprobs_reference).

        Returns:
            (batch, seq) token-level distillation advantages (stop-gradient).
        """
        if teacher_logprobs is None:
            raise ValueError("OPD advantage estimator requires teacher_logprobs")
        if logprobs_policy is None:
            raise ValueError(
                "OPD advantage estimator requires logprobs_policy (student "
                "prev_logprobs); the config must not zero prev_logprobs"
            )

        # Â_MOPD,t = sg[log π_teacher - log π_student]  (Eq. 8)
        distill_advantages = (teacher_logprobs - logprobs_policy).detach()
        advantages = distill_advantages * mask

        self._compute_metrics(distill_advantages, advantages, mask)
        return advantages

    def _compute_metrics(self, distill_advantages, advantages, mask) -> None:
        """Compute OPD logging metrics into self.last_metrics."""
        valid_bool = mask.bool()
        distill_valid = torch.masked_select(distill_advantages, valid_bool)
        adv_valid = torch.masked_select(advantages, valid_bool)

        distill_mean = distill_valid.mean().item() if distill_valid.numel() > 0 else 0.0
        adv_mean = adv_valid.mean().item() if adv_valid.numel() > 0 else 0.0
        adv_std = adv_valid.std().item() if adv_valid.numel() > 1 else 0.0

        self.last_metrics = {
            "on_policy_distillation/teacher_student_logprob_gap_mean": distill_mean,
            "on_policy_distillation/adv_mean": adv_mean,
            "on_policy_distillation/adv_std": adv_std,
        }