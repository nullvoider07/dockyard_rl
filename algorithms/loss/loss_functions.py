"""Loss functions for Project Dockyard RL training.

Contains:
  ClippedPGLossFn   — GRPO / PPO / REINFORCE / DAPO / GSPO clipped PG loss
  NLLLossFn         — Negative log-likelihood (SFT auxiliary loss)
  PreferenceLossFn  — Preference-based loss base class
  DPOLossFn         — Direct Preference Optimisation
  DistillationLossFn — Forward / reverse / mixed KL distillation
  DraftCrossEntropyLossFn — Speculative-decoding draft model training loss

DistributedCrossEntropy is an optional dependency imported lazily.
"""

import math
from typing import Any, NotRequired, Optional, TypedDict, TypeVar
import torch
from pydantic import BaseModel
from dockyard_rl.algorithms.loss.interfaces import LossFunction, LossInputType, LossType
from dockyard_rl.algorithms.loss.utils import calculate_kl, masked_mean
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

# Draft model loss
class DraftCrossEntropyLossConfig(TypedDict):
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup]

class DraftCrossEntropyLossDataDict(TypedDict):
    teacher_logits: torch.Tensor
    student_logits: torch.Tensor
    token_mask:     torch.Tensor
    sample_mask:    torch.Tensor
    student_vocab_indices: NotRequired[torch.Tensor]

class DraftCrossEntropyLossFn(LossFunction):
    """Auxiliary soft-target cross-entropy for draft-model training."""

    loss_type  = LossType.TOKEN_LEVEL
    input_type = LossInputType.DRAFT

    def __init__(
        self,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        self.vocab_parallel_group = vocab_parallel_group

    def __call__(  # type: ignore[override]
        self,
        teacher_logits:    torch.Tensor,
        student_logits:    torch.Tensor,
        token_mask:        torch.Tensor,
        data:              BatchedDataDict[DraftCrossEntropyLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> torch.Tensor:
        if self.vocab_parallel_group is not None:
            from dockyard_rl.distributed.model_utils import DistributedCrossEntropy  # type: ignore[import]
            per_token_loss = DistributedCrossEntropy.apply(
                student_logits, teacher_logits, self.vocab_parallel_group, False
            )
        else:
            teacher_probs      = torch.nn.functional.softmax(teacher_logits, dim=-1)
            student_log_probs  = torch.nn.functional.log_softmax(student_logits, dim=-1)
            per_token_loss     = -(teacher_probs * student_log_probs).sum(dim=-1)

        mask = token_mask * data["sample_mask"].unsqueeze(-1)
        return masked_mean(
            per_token_loss,
            mask,
            global_normalization_factor=global_valid_toks,
        )

# Clipped policy-gradient loss (GRPO / PPO / DAPO / GSPO)
class ClippedPGLossConfig(BaseModel, extra="allow"):
    # --- Loss type ---
    disable_ppo_ratio:                bool  = False
    token_level_loss:                 bool  = True
    sequence_level_importance_ratios: bool  = False

    # --- Clipping ---
    ratio_clip_min: float          = 0.2
    ratio_clip_max: float          = 0.2
    ratio_clip_c:   Optional[float] = None  # dual-clip; None to disable

    # --- KL regularisation ---
    reference_policy_kl_penalty:  float          = 0.01
    reference_policy_kl_type:     str            = "k3"
    kl_input_clamp_value:         Optional[float] = 20.0
    kl_output_clamp_value:        Optional[float] = 10.0
    use_kl_in_reward:             bool = False

    # --- Importance sampling ---
    use_importance_sampling_correction: bool          = False
    truncated_importance_sampling_type:  Optional[str] = None
    truncated_importance_sampling_ratio: Optional[float] = None
    truncated_importance_sampling_ratio_min: Optional[float] = None

    # --- On-policy ---
    use_on_policy_kl_approximation: bool = False
    force_on_policy_ratio:          bool = False

class ClippedPGLossDataDict(TypedDict):
    input_ids:                    torch.Tensor
    advantages:                   torch.Tensor
    prev_logprobs:                torch.Tensor
    generation_logprobs:          torch.Tensor
    reference_policy_logprobs:    torch.Tensor
    token_mask:                   torch.Tensor
    sample_mask:                  torch.Tensor
    __extra__:                    Any

class ClippedPGLossFn(LossFunction):
    """Generalised Clipped Policy Gradient loss.

    Implements PPO, GRPO, REINFORCE/RLOO, DAPO, GSPO, and dual-clipping
    from a single configurable loss function.

    Loss formula:
        L(θ) = E_t[ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ]
               - β * KL(π_θ || π_ref)

    where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t).
    """

    input_type = LossInputType.LOGPROB

    def __init__(self, cfg: ClippedPGLossConfig) -> None:
        self.disable_ppo_ratio                   = cfg.disable_ppo_ratio
        self.ratio_clip_min                      = cfg.ratio_clip_min
        self.ratio_clip_max                      = cfg.ratio_clip_max
        self.ratio_clip_c                        = cfg.ratio_clip_c
        self.reference_policy_kl_penalty         = cfg.reference_policy_kl_penalty
        self.reference_policy_kl_type            = cfg.reference_policy_kl_type
        self.kl_input_clamp_value                = cfg.kl_input_clamp_value
        self.kl_output_clamp_value               = cfg.kl_output_clamp_value
        self.use_importance_sampling_correction  = cfg.use_importance_sampling_correction
        self.truncated_importance_sampling_type  = cfg.truncated_importance_sampling_type
        self.truncated_importance_sampling_ratio = cfg.truncated_importance_sampling_ratio
        self.truncated_importance_sampling_ratio_min = cfg.truncated_importance_sampling_ratio_min
        self.use_on_policy_kl_approximation      = cfg.use_on_policy_kl_approximation
        self.force_on_policy_ratio               = cfg.force_on_policy_ratio
        self.sequence_level_importance_ratios    = cfg.sequence_level_importance_ratios
        self.loss_type = (
            LossType.TOKEN_LEVEL if cfg.token_level_loss else LossType.SEQUENCE_LEVEL
        )

        if self.sequence_level_importance_ratios:
            assert self.loss_type == LossType.SEQUENCE_LEVEL, (
                "sequence_level_importance_ratios is mutually exclusive with token_level_loss"
            )

        if self.truncated_importance_sampling_type is not None:
            assert self.use_importance_sampling_correction, (
                "truncated IS requires use_importance_sampling_correction=True"
            )
            assert self.truncated_importance_sampling_type in ("tis", "icepop", "seq-mask-tis"), (
                f"Invalid truncated IS type: {self.truncated_importance_sampling_type!r}"
            )
            assert (
                self.truncated_importance_sampling_ratio is not None
                and self.truncated_importance_sampling_ratio > 0
            ), "truncated_importance_sampling_ratio must be positive"
            if self.truncated_importance_sampling_type in ("icepop", "seq-mask-tis"):
                assert self.truncated_importance_sampling_ratio_min is not None, (
                    "truncated_importance_sampling_ratio_min required for icepop / seq-mask-tis"
                )
            if self.truncated_importance_sampling_type == "seq-mask-tis":
                assert not self.sequence_level_importance_ratios, (
                    "seq-mask-tis is incompatible with sequence_level_importance_ratios=True"
                )

    def __call__(  # type: ignore[override]
        self,
        next_token_logprobs: torch.Tensor,
        data:                BatchedDataDict[ClippedPGLossDataDict],
        global_valid_seqs:   torch.Tensor,
        global_valid_toks:   torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        curr_logprobs     = next_token_logprobs
        token_mask        = data["token_mask"][:, 1:]
        sample_mask       = data["sample_mask"]
        advantages        = data["advantages"][:, 1:]
        prev_logprobs     = (
            None if self.force_on_policy_ratio
            else data["prev_logprobs"][:, 1:]
        )
        generation_logprobs = data["generation_logprobs"][:, 1:]

        reference_policy_logprobs: torch.Tensor | None = None
        curr_logprobs_unfiltered:  torch.Tensor | None = None
        if self.reference_policy_kl_penalty != 0:
            reference_policy_logprobs    = data["reference_policy_logprobs"][:, 1:]
            curr_logprobs_unfiltered     = data.get(
                "curr_logprobs_unfiltered", curr_logprobs
            )

        mask = token_mask * sample_mask.unsqueeze(-1)

        if self.force_on_policy_ratio:
            prev_logprobs = curr_logprobs.detach()

        # --- Staleness metrics ---
        assert prev_logprobs is not None
        lp_error = torch.abs(generation_logprobs - prev_logprobs)  # noqa: F841
        mult_prob_error = masked_mean(
            torch.exp(lp_error * mask),
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        gen_kl_error = masked_mean(
            calculate_kl(
                logprobs=generation_logprobs,
                logprobs_reference=prev_logprobs,
                kl_type=self.reference_policy_kl_type,
                input_clamp_value=None,
                output_clamp_value=None,
            ),
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        policy_kl_error = masked_mean(
            calculate_kl(
                logprobs=prev_logprobs,
                logprobs_reference=generation_logprobs,
                kl_type=self.reference_policy_kl_type,
                input_clamp_value=None,
                output_clamp_value=None,
            ),
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        log_mixture = torch.log(
            0.5 * torch.exp(prev_logprobs) + 0.5 * torch.exp(generation_logprobs)
        )
        kl_prev_to_mixture = (
            torch.exp(prev_logprobs - log_mixture) - (prev_logprobs - log_mixture) - 1
        )
        kl_gen_to_mixture = (
            torch.exp(generation_logprobs - log_mixture)
            - (generation_logprobs - log_mixture)
            - 1
        )
        js_divergence_error = masked_mean(
            0.5 * kl_prev_to_mixture + 0.5 * kl_gen_to_mixture,
            mask,
            global_normalization_factor=global_valid_toks,
        ).item()

        # --- KL regularisation ---
        if self.reference_policy_kl_penalty != 0:
            assert curr_logprobs_unfiltered is not None
            assert reference_policy_logprobs is not None
            if self.use_on_policy_kl_approximation:
                kl_importance_weights = torch.exp(
                    curr_logprobs_unfiltered - generation_logprobs
                ).detach()
                kl_importance_weights = torch.nan_to_num(
                    kl_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
                )
            else:
                kl_importance_weights = torch.ones_like(curr_logprobs_unfiltered)

            kl = (
                kl_importance_weights
                * self.reference_policy_kl_penalty
                * calculate_kl(
                    logprobs=curr_logprobs_unfiltered,
                    logprobs_reference=reference_policy_logprobs,
                    kl_type=self.reference_policy_kl_type,
                    input_clamp_value=self.kl_input_clamp_value,
                    output_clamp_value=self.kl_output_clamp_value,
                )
            )

            if self.loss_type == LossType.TOKEN_LEVEL:
                kl = masked_mean(kl, mask, global_normalization_factor=global_valid_toks)
            else:
                kl = masked_mean(
                    masked_mean(kl, token_mask, dim=-1),
                    sample_mask,
                    global_normalization_factor=global_valid_seqs,
                )
        else:
            kl = torch.tensor(0.0)

        # --- Probability ratio ---
        if self.force_on_policy_ratio:
            log_ratios = curr_logprobs - curr_logprobs.detach()
            ratios = log_ratios.exp()
            ratios_clamped = ratios
        elif not self.disable_ppo_ratio:
            log_ratios = curr_logprobs - prev_logprobs
            if self.sequence_level_importance_ratios:
                seq_log_ratio_mean = masked_mean(log_ratios, token_mask, dim=-1).unsqueeze(-1)
                seq_ratio = seq_log_ratio_mean.exp()
                ratios = seq_ratio.repeat(1, advantages.shape[1])
            else:
                ratios = log_ratios.exp()
            ratios_clamped = ratios.clamp(
                1.0 - self.ratio_clip_min, 1.0 + self.ratio_clip_max
            )
        else:
            ratios = curr_logprobs
            ratios_clamped = curr_logprobs

        loss1 = -advantages * ratios
        loss2 = -advantages * ratios_clamped
        clip_loss = torch.max(loss1, loss2)

        if self.ratio_clip_c is not None:
            assert self.ratio_clip_c > 1, (
                f"ratio_clip_c must exceed 1, got {self.ratio_clip_c}"
            )
            loss3 = -advantages * self.ratio_clip_c
            clip_loss = torch.where(
                advantages < 0, torch.min(clip_loss, loss3), clip_loss
            )

        # --- Off-policy IS correction ---
        _is_filter_metrics: dict = {}
        if self.sequence_level_importance_ratios:
            seq_lp_diff = ((prev_logprobs - generation_logprobs) * mask).sum(dim=-1)
            actor_importance_weights_expanded = torch.exp(seq_lp_diff).detach()
            actor_importance_weights_expanded = torch.nan_to_num(
                actor_importance_weights_expanded, nan=0.0, posinf=0.0, neginf=0.0
            ).unsqueeze(-1)
        else:
            actor_importance_weights_expanded = torch.exp(prev_logprobs - generation_logprobs)
            actor_importance_weights_expanded = torch.nan_to_num(
                actor_importance_weights_expanded, nan=0.0, posinf=0.0, neginf=0.0
            )

        if self.truncated_importance_sampling_ratio is not None:
            if self.truncated_importance_sampling_type == "tis":
                token_in_bounds = (
                    actor_importance_weights_expanded <= self.truncated_importance_sampling_ratio
                )
                _is_filter_metrics = {
                    "is_oob_ratio": 1.0 - masked_mean(
                        token_in_bounds.float(), mask,
                        global_normalization_factor=global_valid_toks,
                    ).item(),
                }
                actor_importance_weights_expanded = torch.clamp(
                    actor_importance_weights_expanded,
                    max=self.truncated_importance_sampling_ratio,
                )
            elif self.truncated_importance_sampling_type == "icepop":
                assert self.truncated_importance_sampling_ratio_min is not None
                token_kept_mask = (
                    actor_importance_weights_expanded >= self.truncated_importance_sampling_ratio_min
                ) & (
                    actor_importance_weights_expanded <= self.truncated_importance_sampling_ratio
                )
                _is_filter_metrics = {
                    "is_oob_ratio": 1.0 - masked_mean(
                        token_kept_mask.float(), mask,
                        global_normalization_factor=global_valid_toks,
                    ).item(),
                }
                actor_importance_weights_expanded = torch.where(
                    token_kept_mask,
                    actor_importance_weights_expanded,
                    torch.zeros_like(actor_importance_weights_expanded),
                )
            elif self.truncated_importance_sampling_type == "seq-mask-tis":
                assert self.truncated_importance_sampling_ratio_min is not None
                log_is_ratio = torch.nan_to_num(
                    prev_logprobs - generation_logprobs, nan=0.0, posinf=0.0, neginf=0.0
                )
                seq_log_is_ratio_mean = masked_mean(log_is_ratio, token_mask, dim=-1)
                seq_geomean_is_ratio  = torch.exp(seq_log_is_ratio_mean).detach()
                seq_kept_mask = (
                    seq_geomean_is_ratio >= self.truncated_importance_sampling_ratio_min
                ) & (
                    seq_geomean_is_ratio <= self.truncated_importance_sampling_ratio
                )
                seq_kept_mask_f = seq_kept_mask.float()
                _is_filter_metrics = {
                    "is_oob_ratio": 1.0 - masked_mean(
                        seq_kept_mask_f, sample_mask,
                        global_normalization_factor=global_valid_seqs,
                    ).item(),
                }
                actor_importance_weights_expanded = (
                    actor_importance_weights_expanded * seq_kept_mask_f.unsqueeze(-1)
                )
            else:
                raise ValueError(
                    f"Invalid truncated IS type: {self.truncated_importance_sampling_type!r}"
                )

        actor_importance_weights = actor_importance_weights_expanded
        del actor_importance_weights_expanded

        importance_weights_to_use = (
            actor_importance_weights if self.use_importance_sampling_correction
            else torch.ones_like(prev_logprobs)
        )

        if self.loss_type == LossType.TOKEN_LEVEL:
            actor_loss = masked_mean(
                importance_weights_to_use * clip_loss,
                mask,
                global_normalization_factor=global_valid_toks,
            )
        else:
            actor_loss = masked_mean(
                masked_mean(importance_weights_to_use * clip_loss, token_mask, dim=-1),
                sample_mask,
                global_normalization_factor=global_valid_seqs,
            )

        if self.sequence_level_importance_ratios:
            sample_importance_ratio = masked_mean(
                actor_importance_weights, sample_mask,
                global_normalization_factor=global_valid_seqs,
            )
        else:
            sample_importance_ratio = masked_mean(
                actor_importance_weights, mask,
                global_normalization_factor=global_valid_toks,
            )

        with torch.no_grad():
            seq_entropy_approx = -masked_mean(
                torch.exp(curr_logprobs - generation_logprobs) * curr_logprobs,
                mask,
                global_normalization_factor=global_valid_toks,
            )

        loss = actor_loss + kl

        with torch.no_grad():
            probs_ratio         = masked_mean(ratios.detach(), mask, global_normalization_factor=global_valid_toks).item()
            probs_ratio_clamped = masked_mean(ratios_clamped.detach(), mask, global_normalization_factor=global_valid_toks).item()
            masked_ratios         = ratios.detach()[mask.bool()]
            masked_ratios_clamped = ratios_clamped.detach()[mask.bool()]
            if masked_ratios.numel() > 0:
                probs_ratio_min          = masked_ratios.min().item()
                probs_ratio_max          = masked_ratios.max().item()
                probs_ratio_clamped_min  = masked_ratios_clamped.min().item()
                probs_ratio_clamped_max  = masked_ratios_clamped.max().item()
            else:
                probs_ratio_min = probs_ratio_max = float("inf")
                probs_ratio_clamped_min = probs_ratio_clamped_max = float("inf")

        return (
            loss,
            {
                "loss":                    loss.item(),
                "probs_ratio":             probs_ratio,
                "probs_ratio_clamped":     probs_ratio_clamped,
                "probs_ratio_min":         probs_ratio_min,
                "probs_ratio_max":         probs_ratio_max,
                "probs_ratio_clamped_min": probs_ratio_clamped_min,
                "probs_ratio_clamped_max": probs_ratio_clamped_max,
                "kl_penalty":              kl.item() / self.reference_policy_kl_penalty if kl else 0,
                "token_mult_prob_error":   mult_prob_error,
                "gen_kl_error":            gen_kl_error,
                "policy_kl_error":         policy_kl_error,
                "js_divergence_error":     js_divergence_error,
                "sampling_importance_ratio": sample_importance_ratio.item(),
                "num_valid_samples":       sample_mask.sum().item(),
                "approx_entropy":          seq_entropy_approx.item(),
                **_is_filter_metrics,
            },
        )

# NLL loss
class NLLLossFn(LossFunction):
    """Negative Log-Likelihood loss (SFT auxiliary)."""

    loss_type  = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __init__(self, use_linear_ce_fusion: bool = False) -> None:
        self.use_linear_ce_fusion = use_linear_ce_fusion

    def __call__(  # type: ignore[override]
        self,
        next_token_logprobs: torch.Tensor,
        data:                BatchedDataDict[Any],
        global_valid_seqs:   torch.Tensor | None,
        global_valid_toks:   torch.Tensor,
        dpo_loss:            bool = False,
        dpo_average_log_probs: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        token_mask  = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        mask        = token_mask * sample_mask.unsqueeze(-1)

        if dpo_loss:
            num_unmasked_tokens = torch.sum(mask, -1)
            loss = -torch.sum(next_token_logprobs * mask, dim=-1)
            if dpo_average_log_probs:
                loss = loss / num_unmasked_tokens.clamp(min=1)
        else:
            loss = -masked_mean(
                next_token_logprobs, mask,
                global_normalization_factor=global_valid_toks,
            )

        return loss, {
            "loss":                  loss.item() if loss.ndim == 0 else loss,
            "num_unmasked_tokens":   mask.sum().item(),
            "num_valid_samples":     sample_mask.sum().item(),
        }

# Preference losses (DPO, IPO, etc.)
class PreferenceLossDataDict(TypedDict):
    input_ids:   torch.Tensor
    token_mask:  torch.Tensor
    sample_mask: torch.Tensor

class PreferenceLossFn(LossFunction):
    """Base class for preference-based losses (DPO, IPO, etc.)."""

    loss_type  = LossType.SEQUENCE_LEVEL
    input_type = LossInputType.LOGIT

    def split_output_tensor(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tensor[::2], tensor[1::2]

    def _preference_loss(
        self,
        rewards:           torch.Tensor,
        sample_mask:       torch.Tensor,
        global_valid_seqs: torch.Tensor,
        beta:              float = 1.0,
        *,
        extra_margin:      Optional[torch.Tensor] = None,
        pre_sigmoid_offset: Optional[torch.Tensor] = None,
        label_smoothing:   float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core preference loss on the reward delta.

        ``extra_margin`` (per-chosen-sample, shape ``[batch/2]``) is added to the
        reward delta *before* the β scaling — DPOP uses it to subtract its
        positive-penalty term (λ is canonically inside β). ``pre_sigmoid_offset``
        is added *after* the β scaling — R-DPO uses it for the length penalty
        (α is canonically outside β). ``label_smoothing`` ε mixes the flipped
        target (cDPO). All default to identity values, so the base DPO path is
        byte-identical when none is supplied.
        """
        rewards_chosen, rewards_rejected = self.split_output_tensor(rewards)
        rewards_delta    = rewards_chosen - rewards_rejected
        if extra_margin is not None:
            rewards_delta = rewards_delta + extra_margin
        logits = beta * rewards_delta
        if pre_sigmoid_offset is not None:
            logits = logits + pre_sigmoid_offset
        if label_smoothing > 0.0:
            per_sample_loss = (
                -(
                    (1.0 - label_smoothing) * torch.nn.functional.logsigmoid(logits)
                    + label_smoothing * torch.nn.functional.logsigmoid(-logits)
                )
                * sample_mask[::2]
            )
        else:
            per_sample_loss = (
                -torch.nn.functional.logsigmoid(logits) * sample_mask[::2]
            )
        return (
            masked_mean(per_sample_loss, sample_mask[::2], global_normalization_factor=global_valid_seqs / 2),
            masked_mean(rewards_chosen > rewards_rejected, sample_mask[::2], global_normalization_factor=global_valid_seqs / 2),
            masked_mean(rewards_chosen,  sample_mask[::2], global_normalization_factor=global_valid_seqs / 2),
            masked_mean(rewards_rejected, sample_mask[1::2], global_normalization_factor=global_valid_seqs / 2),
        )

    def __call__(  # type: ignore[override]
        self,
        logits:            torch.Tensor,
        data:              BatchedDataDict[PreferenceLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sample_mask = data["sample_mask"]
        rewards     = logits.squeeze(-1)
        preference_loss, accuracy, rewards_chosen_mean, rewards_rejected_mean = (
            self._preference_loss(rewards, sample_mask, global_valid_seqs)
        )
        num_valid_samples = sample_mask.sum() / 2
        return preference_loss, {
            "loss":                  preference_loss.item(),
            "accuracy":              accuracy.item(),
            "rewards_chosen_mean":   rewards_chosen_mean.item(),
            "rewards_rejected_mean": rewards_rejected_mean.item(),
            "num_valid_samples":     num_valid_samples.item(),
        }

# DPO loss
class DPOLossConfig(TypedDict):
    reference_policy_kl_penalty:  float
    preference_loss_weight:        float
    sft_loss_weight:               float
    preference_average_log_probs:  bool
    sft_average_log_probs:         bool
    # Variant selection + variant-specific hyperparameters (optional; identity defaults).
    loss_variant:                  NotRequired[str]    # dpo | dpop | cdpo | rdpo | ipo | kto
    label_smoothing:               NotRequired[float]  # cDPO ε; 0 ⇒ DPO
    dpop_lambda:                   NotRequired[float]  # DPOP λ; 0 ⇒ DPO
    length_penalty:                NotRequired[float]  # R-DPO α; 0 ⇒ DPO
    desirable_weight:              NotRequired[float]  # KTO λ_D (desirable examples)
    undesirable_weight:            NotRequired[float]  # KTO λ_U (undesirable examples)
    simpo_gamma:                   NotRequired[float]  # SimPO γ target margin (reference-free)
    orpo_lambda:                   NotRequired[float]  # ORPO λ odds-ratio weight (reference-free)

class DPOLossDataDict(TypedDict):
    input_ids:                  torch.Tensor
    reference_policy_logprobs:  torch.Tensor
    token_mask:                 torch.Tensor
    sample_mask:                torch.Tensor

class DPOLossFn(PreferenceLossFn):
    """Direct Preference Optimisation loss."""

    loss_type  = LossType.SEQUENCE_LEVEL
    input_type = LossInputType.LOGPROB

    # Whether the variant needs frozen-reference logprobs precomputed
    # (add_ref_logprobs_to_data consults this via the variant registry).
    reference_free = False

    # Number of sequences each preference datum expands into. The DPO family is
    # paired (chosen+rejected interleaved → 2); KTO is unpaired (→ 1). The driver
    # reads this to size global/micro batches; default preserves the paired path.
    sequences_per_datum = 2

    def __init__(self, cfg: DPOLossConfig, use_linear_ce_fusion: bool = False) -> None:
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.preference_loss_weight      = cfg["preference_loss_weight"]
        self.sft_loss_weight             = cfg["sft_loss_weight"]
        self.preference_average_log_probs = cfg["preference_average_log_probs"]
        self.sft_average_log_probs       = cfg["sft_average_log_probs"]
        self.use_linear_ce_fusion        = use_linear_ce_fusion
        self.sft_loss                    = NLLLossFn(use_linear_ce_fusion=use_linear_ce_fusion)
        # Variant hyperparameters; identity values reduce to vanilla DPO.
        self.label_smoothing             = 0.0
        self.dpop_lambda                 = 0.0
        self.length_penalty              = 0.0

    def _dpop_extra_margin(
        self,
        next_token_logprobs: torch.Tensor,
        ref_logprobs:        torch.Tensor,
        token_mask:          torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """DPOP positive penalty as an additive (negative) margin on the chosen rows.

        Returns ``-λ·max(0, s_ref_chosen − s_policy_chosen)`` per chosen sample, or
        ``None`` when λ=0 (so the reward delta is untouched and the loss is exactly DPO).
        """
        if self.dpop_lambda == 0.0:
            return None
        policy_sum = (next_token_logprobs * token_mask).sum(-1)
        ref_sum    = (ref_logprobs * token_mask).sum(-1)
        if self.preference_average_log_probs:
            denom      = token_mask.sum(-1).clamp(min=1)
            policy_sum = policy_sum / denom
            ref_sum    = ref_sum / denom
        penalty = torch.clamp(ref_sum - policy_sum, min=0.0)
        chosen_penalty, _ = self.split_output_tensor(penalty)
        return -self.dpop_lambda * chosen_penalty

    def _rdpo_pre_sigmoid_offset(
        self,
        token_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """R-DPO length penalty as a post-β-scaling offset on the chosen rows.

        Returns ``-α·(|y_chosen| − |y_rejected|)`` per pair, or ``None`` when α=0
        (so the loss is exactly DPO). ``token_mask`` is the response mask already
        sliced to ``[:, 1:]``; its row sum is the response length |y|.
        """
        if self.length_penalty == 0.0:
            return None
        lengths = token_mask.sum(-1)
        len_chosen, len_rejected = self.split_output_tensor(lengths)
        return -self.length_penalty * (len_chosen - len_rejected)

    def _dpo_loss(
        self,
        next_token_logprobs: torch.Tensor,
        data:                BatchedDataDict[DPOLossDataDict],
        global_valid_seqs:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_mask  = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        ref_logprobs = data["reference_policy_logprobs"][:, :-1]
        diff    = (next_token_logprobs - ref_logprobs) * token_mask
        rewards = diff.sum(-1)
        if self.preference_average_log_probs:
            rewards = rewards / token_mask.sum(-1).clamp(min=1)
        extra_margin = self._dpop_extra_margin(next_token_logprobs, ref_logprobs, token_mask)
        pre_sigmoid_offset = self._rdpo_pre_sigmoid_offset(token_mask)
        return self._preference_loss(
            rewards, sample_mask, global_valid_seqs, self.reference_policy_kl_penalty,
            extra_margin=extra_margin,
            pre_sigmoid_offset=pre_sigmoid_offset,
            label_smoothing=self.label_smoothing,
        )

    def __call__(  # type: ignore[override]
        self,
        next_token_logprobs: torch.Tensor,
        data:                BatchedDataDict[DPOLossDataDict],
        global_valid_seqs:   torch.Tensor,
        global_valid_toks:   torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sft_loss_chosen = torch.tensor(0.0)
        if self.sft_loss_weight > 0:
            assert global_valid_toks is not None
            sft_loss, _ = self.sft_loss(
                next_token_logprobs, data,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
                dpo_loss=True,
                dpo_average_log_probs=self.sft_average_log_probs,
            )
            sft_chosen, _ = self.split_output_tensor(sft_loss)
            sft_loss_chosen = masked_mean(
                sft_chosen, data["sample_mask"][::2],
                global_normalization_factor=global_valid_seqs / 2,
            )

        preference_loss, accuracy, rewards_chosen_mean, rewards_rejected_mean = (
            self._dpo_loss(next_token_logprobs, data, global_valid_seqs)
        )
        dpo_loss = (
            self.sft_loss_weight * sft_loss_chosen
            + self.preference_loss_weight * preference_loss
        )
        num_valid_samples = data["sample_mask"].sum() / 2
        return dpo_loss, {
            "loss":                  dpo_loss.item(),
            "sft_loss":              sft_loss_chosen.item(),
            "preference_loss":       preference_loss.item(),
            "accuracy":              accuracy.item(),
            "rewards_chosen_mean":   rewards_chosen_mean.item(),
            "rewards_rejected_mean": rewards_rejected_mean.item(),
            "num_valid_samples":     num_valid_samples.item(),
        }

# Preference loss variants (DPOP, cDPO, R-DPO, IPO, KTO).
# All reuse DPOLossConfig — the variant hyperparameters (label_smoothing,
# dpop_lambda, length_penalty, desirable_weight, undesirable_weight) are optional
# keys on it with identity/neutral defaults.
class CDPOLossFn(DPOLossFn):
    """Conservative DPO (cDPO) — label-smoothed preference loss.

    Mixes the flipped target with weight ε to tolerate noisy / mislabelled
    preferences:  L = −[(1−ε)·logσ(βΔ) + ε·logσ(−βΔ)].  Reduces to DPO at ε=0.
    """

    def __init__(self, cfg: DPOLossConfig, use_linear_ce_fusion: bool = False) -> None:
        super().__init__(cfg, use_linear_ce_fusion=use_linear_ce_fusion)
        self.label_smoothing = float(cfg.get("label_smoothing", 0.0))
        assert 0.0 <= self.label_smoothing < 0.5, (
            f"cDPO label_smoothing must be in [0, 0.5), got {self.label_smoothing}"
        )

class DPOPLossFn(DPOLossFn):
    """DPO-positive (DPOP) — adds a positive penalty that resists the chosen
    log-probability dropping below the reference.

        L = −logσ(β·(Δ − λ·max(0, s_ref_chosen − s_policy_chosen)))

    Mitigates the failure where both chosen and rejected log-probs fall (a
    capability regression). Reduces to DPO at λ=0.
    """

    def __init__(self, cfg: DPOLossConfig, use_linear_ce_fusion: bool = False) -> None:
        super().__init__(cfg, use_linear_ce_fusion=use_linear_ce_fusion)
        self.dpop_lambda = float(cfg.get("dpop_lambda", 0.0))
        assert self.dpop_lambda >= 0.0, (
            f"DPOP dpop_lambda must be non-negative, got {self.dpop_lambda}"
        )

class RDPOLossFn(DPOLossFn):
    """Length-regularised DPO (R-DPO) — subtracts an explicit length penalty
    from the implicit reward to disentangle preference quality from response
    length.

        L = −logσ(β·Δ − α·(|y_chosen| − |y_rejected|))

    The α penalty is outside the β scaling (per Park et al. 2024). Reduces to
    DPO at α=0.
    """

    def __init__(self, cfg: DPOLossConfig, use_linear_ce_fusion: bool = False) -> None:
        super().__init__(cfg, use_linear_ce_fusion=use_linear_ce_fusion)
        self.length_penalty = float(cfg.get("length_penalty", 0.0))
        assert self.length_penalty >= 0.0, (
            f"R-DPO length_penalty must be non-negative, got {self.length_penalty}"
        )

class IPOLossFn(DPOLossFn):
    """Identity Preference Optimisation (IPO) — replaces the logsigmoid with a
    squared loss around a target margin, regularising toward the reference
    rather than over-fitting deterministic preferences.

        L = (Δ − 1/(2τ))²,   τ = reference_policy_kl_penalty

    Distinct loss family — does not reduce to DPO at any hyperparameter.
    """

    def _preference_loss(  # type: ignore[override]
        self,
        rewards:           torch.Tensor,
        sample_mask:       torch.Tensor,
        global_valid_seqs: torch.Tensor,
        beta:              float = 1.0,
        *,
        extra_margin:      Optional[torch.Tensor] = None,
        pre_sigmoid_offset: Optional[torch.Tensor] = None,
        label_smoothing:   float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # IPO uses only the reward delta; DPOP/R-DPO/cDPO modifiers do not apply.
        assert beta > 0.0, "IPO requires reference_policy_kl_penalty (τ) > 0"
        rewards_chosen, rewards_rejected = self.split_output_tensor(rewards)
        rewards_delta = rewards_chosen - rewards_rejected
        target = 1.0 / (2.0 * beta)
        per_sample_loss = ((rewards_delta - target) ** 2) * sample_mask[::2]
        return (
            masked_mean(per_sample_loss, sample_mask[::2], global_normalization_factor=global_valid_seqs / 2),
            masked_mean(rewards_chosen > rewards_rejected, sample_mask[::2], global_normalization_factor=global_valid_seqs / 2),
            masked_mean(rewards_chosen,  sample_mask[::2], global_normalization_factor=global_valid_seqs / 2),
            masked_mean(rewards_rejected, sample_mask[1::2], global_normalization_factor=global_valid_seqs / 2),
        )

class KTOLossFn(DPOLossFn):
    """Kahneman-Tversky Optimisation (KTO) — unpaired preference loss.

    Operates on a batch of independently-labelled examples (desirable or
    undesirable) rather than chosen/rejected pairs. Per example, with implicit
    reward (logratio) r = Σ_t (logπ − logπ_ref)·mask and a detached KL reference
    point z:

        desirable:    λ_D · (1 − σ(β·(r − z)))
        undesirable:  λ_U · (1 − σ(β·(z − r)))

    z is the policy↔reference KL estimated from mismatched completions through
    the *current* policy; it is supplied at train time via ``data["kto_reference_kl"]``
    (worker-computed, detached, clamped ≥ 0). When absent it defaults to 0 (no KL
    anchor) — the pre-worker behaviour. Consumes ``preference_label`` (1 desirable
    / 0 undesirable) from the unpaired ``kto_collate_fn``.
    """

    sequences_per_datum = 1

    def __init__(self, cfg: DPOLossConfig, use_linear_ce_fusion: bool = False) -> None:
        super().__init__(cfg, use_linear_ce_fusion=use_linear_ce_fusion)
        self.desirable_weight   = float(cfg.get("desirable_weight", 1.0))
        self.undesirable_weight = float(cfg.get("undesirable_weight", 1.0))
        assert self.desirable_weight >= 0.0 and self.undesirable_weight >= 0.0, (
            "KTO desirable_weight / undesirable_weight must be non-negative"
        )

    def __call__(  # type: ignore[override]
        self,
        next_token_logprobs: torch.Tensor,
        data:                BatchedDataDict[Any],
        global_valid_seqs:   torch.Tensor,
        global_valid_toks:   torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        token_mask  = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        ref_logprobs = data["reference_policy_logprobs"][:, :-1]
        label = data["preference_label"]  # [B], 1.0 desirable / 0.0 undesirable

        beta = self.reference_policy_kl_penalty
        logratios = ((next_token_logprobs - ref_logprobs) * token_mask).sum(-1)

        kl_raw = data.get("kto_reference_kl", None)
        if kl_raw is None:
            kl = torch.zeros((), dtype=logratios.dtype, device=logratios.device)
        else:
            kl = torch.as_tensor(
                kl_raw, dtype=logratios.dtype, device=logratios.device
            ).detach()

        desirable_term   = 1.0 - torch.sigmoid(beta * (logratios - kl))
        undesirable_term = 1.0 - torch.sigmoid(beta * (kl - logratios))
        per_example = (
            label * self.desirable_weight * desirable_term
            + (1.0 - label) * self.undesirable_weight * undesirable_term
        )
        loss = masked_mean(
            per_example, sample_mask, global_normalization_factor=global_valid_seqs
        )

        with torch.no_grad():
            rewards = beta * logratios
            desirable_mask   = sample_mask * label
            undesirable_mask = sample_mask * (1.0 - label)
            rewards_desirable_mean   = masked_mean(rewards, desirable_mask)
            rewards_undesirable_mean = masked_mean(rewards, undesirable_mask)

        return loss, {
            "loss":                     loss.item(),
            "kl":                       kl.item(),
            "rewards_desirable_mean":   rewards_desirable_mean.item(),
            "rewards_undesirable_mean": rewards_undesirable_mean.item(),
            "rewards_margin":           (rewards_desirable_mean - rewards_undesirable_mean).item(),
            "num_desirable":            desirable_mask.sum().item(),
            "num_undesirable":          undesirable_mask.sum().item(),
            "num_valid_samples":        sample_mask.sum().item(),
        }

# Reference-free preference loss variants (SimPO, ORPO).
def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 − exp(x)) for x ≤ 0 (Mächler 2012).

    Used by ORPO's odds computation. Assumes x < 0 (strictly negative average
    log-probabilities); at x = 0 it is −∞ by definition.
    """
    return torch.where(
        x > -math.log(2.0),
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )

class SimPOLossFn(DPOLossFn):
    """SimPO — reference-free, length-normalised preference loss with a target margin.

    Reward is the average log-probability of the response (no reference model):
    s = (1/|y|) Σ_t logπ(y_t). The loss is

        L = −logσ(β·(s_chosen − s_rejected) − γ)

    where γ = ``simpo_gamma`` is the target reward margin. Reference-free: needs
    no frozen-reference logprobs.
    """

    reference_free = True

    def __init__(self, cfg: DPOLossConfig, use_linear_ce_fusion: bool = False) -> None:
        super().__init__(cfg, use_linear_ce_fusion=use_linear_ce_fusion)
        self.simpo_gamma = float(cfg.get("simpo_gamma", 0.0))
        assert self.simpo_gamma >= 0.0, (
            f"SimPO simpo_gamma must be non-negative, got {self.simpo_gamma}"
        )

    def _dpo_loss(  # type: ignore[override]
        self,
        next_token_logprobs: torch.Tensor,
        data:                BatchedDataDict[Any],
        global_valid_seqs:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_mask  = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        # SimPO reward: length-normalised average log-prob, no reference model.
        rewards = (next_token_logprobs * token_mask).sum(-1) / token_mask.sum(-1).clamp(min=1)
        offset: Optional[torch.Tensor] = None
        if self.simpo_gamma != 0.0:
            offset = rewards.new_full((rewards.shape[0] // 2,), -self.simpo_gamma)
        return self._preference_loss(
            rewards, sample_mask, global_valid_seqs, self.reference_policy_kl_penalty,
            pre_sigmoid_offset=offset,
            label_smoothing=self.label_smoothing,
        )

class ORPOLossFn(DPOLossFn):
    """ORPO — reference-free monolithic objective: SFT NLL plus a log-odds-ratio term.

    With length-normalised average log-probabilities s = (1/|y|) Σ logπ:

        log_odds = (s_chosen − s_rejected) − (log1mexp(s_chosen) − log1mexp(s_rejected))
        L        = −s_chosen  +  λ·(−logσ(log_odds))

    where λ = ``orpo_lambda`` weights the odds-ratio term against the SFT loss.
    Reference-free; the SFT term is the length-normalised NLL on the chosen
    response (sequence-weighted, matching the repo's DPO SFT-aux convention).
    """

    reference_free = True

    def __init__(self, cfg: DPOLossConfig, use_linear_ce_fusion: bool = False) -> None:
        super().__init__(cfg, use_linear_ce_fusion=use_linear_ce_fusion)
        self.orpo_lambda = float(cfg.get("orpo_lambda", 0.1))
        assert self.orpo_lambda >= 0.0, (
            f"ORPO orpo_lambda must be non-negative, got {self.orpo_lambda}"
        )

    def __call__(  # type: ignore[override]
        self,
        next_token_logprobs: torch.Tensor,
        data:                BatchedDataDict[Any],
        global_valid_seqs:   torch.Tensor,
        global_valid_toks:   torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        token_mask  = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        logp_avg = (next_token_logprobs * token_mask).sum(-1) / token_mask.sum(-1).clamp(min=1)
        logp_chosen, logp_rejected = self.split_output_tensor(logp_avg)

        log_odds = (logp_chosen - logp_rejected) - (
            _log1mexp(logp_chosen) - _log1mexp(logp_rejected)
        )
        or_loss    = -torch.nn.functional.logsigmoid(log_odds)
        nll_chosen = -logp_chosen
        per_pair   = nll_chosen + self.orpo_lambda * or_loss

        norm = global_valid_seqs / 2
        loss = masked_mean(per_pair, sample_mask[::2], global_normalization_factor=norm)
        with torch.no_grad():
            accuracy      = masked_mean((logp_chosen > logp_rejected).float(), sample_mask[::2], global_normalization_factor=norm)
            sft_loss_mean = masked_mean(nll_chosen, sample_mask[::2], global_normalization_factor=norm)
            or_loss_mean  = masked_mean(or_loss, sample_mask[::2], global_normalization_factor=norm)
            log_odds_mean = masked_mean(log_odds, sample_mask[::2], global_normalization_factor=norm)
        num_valid_samples = sample_mask.sum() / 2
        return loss, {
            "loss":              loss.item(),
            "sft_loss":          sft_loss_mean.item(),
            "or_loss":           or_loss_mean.item(),
            "log_odds_ratio":    log_odds_mean.item(),
            "accuracy":          accuracy.item(),
            "num_valid_samples": num_valid_samples.item(),
        }

# Preference-loss variant registry + factory
PREFERENCE_LOSS_REGISTRY: dict[str, type[DPOLossFn]] = {
    "dpo":   DPOLossFn,
    "dpop":  DPOPLossFn,
    "cdpo":  CDPOLossFn,
    "rdpo":  RDPOLossFn,
    "ipo":   IPOLossFn,
    "kto":   KTOLossFn,
    "simpo": SimPOLossFn,
    "orpo":  ORPOLossFn,
}

def build_preference_loss(
    variant: Optional[str],
    cfg: DPOLossConfig,
    use_linear_ce_fusion: bool = False,
) -> DPOLossFn:
    """Construct the preference loss for ``dpo.loss_variant``.

    A missing/None variant defaults to vanilla DPO. An unrecognised variant
    raises a clear error rather than silently falling back.
    """
    key = (variant or "dpo").lower()
    try:
        cls = PREFERENCE_LOSS_REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"Unknown/unsupported preference loss_variant {variant!r}. "
            f"Available: {sorted(PREFERENCE_LOSS_REGISTRY)}"
        ) from None
    return cls(cfg, use_linear_ce_fusion=use_linear_ce_fusion)

def is_reference_free(variant: Optional[str]) -> bool:
    """Whether a variant needs no frozen-reference logprobs (SimPO/ORPO, D4).

    Consulted by the DPO driver to skip the reference-logprob stage. Returns
    False for every variant currently registered, so the frozen-ref path is
    unchanged for dpo/dpop/cdpo/rdpo/ipo/kto.
    """
    cls = PREFERENCE_LOSS_REGISTRY.get((variant or "dpo").lower())
    return bool(cls is not None and getattr(cls, "reference_free", False))

# Distillation loss
class DistillationLossConfig(TypedDict):
    kl_type:            str
    mixed_kl_weight:    float
    zero_outside_topk:  bool

class DistillationLossDataDict(TypedDict):
    input_ids:              torch.Tensor
    input_lengths:          torch.Tensor
    token_mask:             torch.Tensor
    sample_mask:            torch.Tensor
    teacher_topk_logits:    torch.Tensor
    teacher_topk_indices:   torch.Tensor

class DistillationLossFn(LossFunction):
    """Forward / reverse / mixed KL distillation loss."""

    loss_type  = LossType.TOKEN_LEVEL
    input_type = LossInputType.DISTILLATION

    def __init__(self, cfg: DistillationLossConfig) -> None:
        self.kl_type            = cfg["kl_type"]
        self.mixed_kl_weight    = cfg["mixed_kl_weight"]
        self.zero_outside_topk  = cfg["zero_outside_topk"]
        self.log_infinitesimal  = -100

        assert self.kl_type in ("forward", "reverse", "mixed"), (
            f"Invalid kl_type {self.kl_type!r}"
        )
        assert 0.0 <= self.mixed_kl_weight <= 1.0, "mixed_kl_weight must be in [0, 1]"

    def __call__(  # type: ignore[override]
        self,
        student_topk_logprobs: torch.Tensor,
        teacher_topk_logprobs: torch.Tensor,
        H_all:                 torch.Tensor | None,
        data:                  DistillationLossDataDict,
        global_valid_seqs:     torch.Tensor,
        global_valid_toks:     torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        student_probs = student_topk_logprobs.exp()
        teacher_probs = teacher_topk_logprobs.exp()

        loss_correction_term = torch.zeros_like(student_probs[..., 0])
        if self.zero_outside_topk and self.kl_type != "forward":
            assert H_all is not None
            H_rest = H_all - (student_probs * student_topk_logprobs).sum(-1)
            P_rest = 1 - student_probs.sum(-1)
            loss_correction_term = H_rest - self.log_infinitesimal * P_rest
            if self.kl_type == "mixed":
                loss_correction_term = loss_correction_term * (1.0 - self.mixed_kl_weight)

        if self.kl_type == "forward":
            per_token_kl = teacher_probs * (teacher_topk_logprobs - student_topk_logprobs)
        elif self.kl_type == "reverse":
            per_token_kl = student_probs * (student_topk_logprobs - teacher_topk_logprobs)
        else:
            kl_forward = teacher_probs * (teacher_topk_logprobs - student_topk_logprobs)
            kl_reverse = student_probs * (student_topk_logprobs - teacher_topk_logprobs)
            per_token_kl = (
                self.mixed_kl_weight * kl_forward
                + (1.0 - self.mixed_kl_weight) * kl_reverse
            )

        per_token_kl = per_token_kl.sum(dim=-1) + loss_correction_term

        if "token_mask" in data and "sample_mask" in data:
            token_mask  = data["token_mask"][:, 1:]
            sample_mask = data["sample_mask"]
            max_len     = per_token_kl.shape[1]
            token_mask  = token_mask[:, :max_len]
            mask        = token_mask * sample_mask.unsqueeze(-1)
            kl_loss     = masked_mean(per_token_kl, mask, global_normalization_factor=global_valid_toks)
        else:
            kl_loss = per_token_kl.mean()

        return kl_loss, {
            "loss":              float(kl_loss.item()) if kl_loss.ndim == 0 else kl_loss,
            "num_valid_samples": data["input_ids"].shape[0],
        }