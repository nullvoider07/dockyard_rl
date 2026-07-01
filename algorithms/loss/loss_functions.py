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
from dockyard_rl.algorithms.x_token.loss_utils import (
    LocalizedAlignment,
    build_exact_token_map,
    ce_label_mask,
    chunk_average_log_probs,
    get_sparse_projection_matrix,
    next_token_accuracy,
    project_student_to_teacher_vocab,
    select_teacher_topk_indices,
    student_next_token_ce,
    valid_chunk_mask,
)
from dockyard_rl.distributed.model_utils import (
    cp_shift_next,
    group_all_reduce_sum,
    vocab_parallel_full_log_softmax,
    vocab_parallel_log_softmax,
)
from dockyard_rl.models.dtensor.parallelize import to_local_if_dtensor

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

    # --- CISPO ---
    # Clipped IS-weight Policy Optimization (arXiv:2506.13585): a REINFORCE-style
    # surrogate with a stop-gradient clipped IS weight. Mutually exclusive with the
    # ratio knobs below; see __init__ asserts.
    use_cispo:                      bool = False

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

        self.use_cispo = cfg.use_cispo
        if self.use_cispo:
            assert not self.disable_ppo_ratio, (
                "use_cispo is incompatible with disable_ppo_ratio; CISPO needs the "
                "pi_theta/pi_theta_old ratio that disable_ppo_ratio removes"
            )
            assert not self.force_on_policy_ratio, (
                "use_cispo is incompatible with force_on_policy_ratio; forcing ratio=1 "
                "removes the clipped IS-weight CISPO optimizes"
            )
            assert not self.sequence_level_importance_ratios, (
                "use_cispo is incompatible with sequence_level_importance_ratios; "
                "CISPO uses token-level importance weights"
            )
            assert self.ratio_clip_c is None, (
                "use_cispo is incompatible with dual clipping (ratio_clip_c); the "
                "dual-clip block runs after CISPO loss assembly and would overwrite it"
            )
            assert self.loss_type == LossType.TOKEN_LEVEL, (
                "use_cispo requires token_level_loss=True (LossType.TOKEN_LEVEL)"
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
            if self.truncated_importance_sampling_ratio_min is not None:
                assert (
                    self.truncated_importance_sampling_ratio_min
                    <= self.truncated_importance_sampling_ratio
                ), (
                    "truncated_importance_sampling_ratio_min must be <= "
                    "truncated_importance_sampling_ratio"
                )
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
            # KL samples are drawn from the optimized policy, so the KL loss must
            # carry the score-function gradient through the sampling probability
            # (https://arxiv.org/abs/2506.09477v1). The on-policy importance ratio
            # exp(curr - generation) provides it; otherwise the straight-through
            # exp(x - x.detach()) has forward value 1 while preserving that gradient.
            if self.use_on_policy_kl_approximation:
                kl_importance_weights = torch.exp(
                    curr_logprobs_unfiltered - generation_logprobs
                )
            else:
                kl_importance_weights = torch.exp(
                    curr_logprobs_unfiltered - curr_logprobs_unfiltered.detach()
                )
            kl_importance_weights = torch.nan_to_num(
                kl_importance_weights, nan=0.0, posinf=0.0, neginf=0.0
            )

            kl = self.reference_policy_kl_penalty * calculate_kl(
                logprobs=curr_logprobs_unfiltered,
                logprobs_reference=reference_policy_logprobs,
                kl_type=self.reference_policy_kl_type,
                input_clamp_value=self.kl_input_clamp_value,
                output_clamp_value=self.kl_output_clamp_value,
                importance_sampling_weights=kl_importance_weights,
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

        if self.use_cispo:
            # CISPO (arXiv:2506.13585): REINFORCE-style surrogate where the clipped
            # IS weight is a stop-gradient scalar, so the gradient flows only through
            # log pi_theta (curr_logprobs). ratio_clip_c is forbidden by __init__.
            clip_loss = -advantages * ratios_clamped.detach() * curr_logprobs
        else:
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
                tis_min = self.truncated_importance_sampling_ratio_min
                if tis_min is None:
                    tis_min = 0.0
                token_in_bounds = (
                    actor_importance_weights_expanded <= self.truncated_importance_sampling_ratio
                ) & (actor_importance_weights_expanded >= tis_min)
                _is_filter_metrics = {
                    "is_oob_ratio": 1.0 - masked_mean(
                        token_in_bounds.float(), mask,
                        global_normalization_factor=global_valid_toks,
                    ).item(),
                }
                actor_importance_weights_expanded = torch.clamp(
                    actor_importance_weights_expanded,
                    min=tis_min,
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


# Cross-tokenizer distillation loss
class CrossTokenizerDistillationLossConfig(TypedDict):
    """Config for (multi-teacher) cross-tokenizer distillation loss.

    ``gold_loss`` / ``xtoken_loss`` and the scalar knobs below are the global
    defaults shared across every teacher; the per-teacher runtime-injected lists
    (``projection_matrix_paths`` etc.) carry one entry per ``teachers[i]`` and
    can override the gold/xtoken flags per teacher in ``kd_loss_mode="sum"``. The
    single-teacher path is just a length-1 list.

    Attributes:
        gold_loss: If True, use the gold-loss formulation: split the vocab into
            an exact-token-mapped *common* set (KL) and an *uncommon* set (L1).
        xtoken_loss: Modifier inside the gold-loss path. If True, relaxes the
            exact-map threshold to ``>= 0.6`` (vs ``== 1.0``) and adds a
            collision-replacement rule. Requires ``gold_loss=True``.
        temperature: Softmax temperature applied symmetrically before KL.
        vocab_topk: Microbatch-global top-k size for the P-KL path
            (``gold_loss=False``). Inert when ``gold_loss=True``.
        uncommon_topk: Cap on the L1 uncommon-tail sort in the gold path.
            Inert when ``gold_loss=False``.
        reverse_kl: If True, KL(student || teacher) instead of KL(teacher ||
            student).
        exact_token_match_only: P-KL path only — if True, only 'is_correct'
            pairs contribute to KL.
        kl_loss_weight: Multiplier on the aggregated KD term in fixed-weight mode.
        ce_loss_scale: Multiplier on the next-token CE term in fixed-weight mode.
        dynamic_loss_scaling: If True, rescale the KD term each step to match the
            detached CE magnitude; ``kl_loss_weight`` / ``ce_loss_scale`` are
            ignored.
        kd_loss_mode: How the per-teacher KD terms combine — ``"sum"`` (weighted
            sum), ``"averaged_logits"`` (convex-average teacher logits then one
            KL), or ``"select_teacher"`` (use only the lowest-CE teacher).
        normalize_teacher_by_vocab: sum-mode only — scale each teacher's KD by
            ``log(V_t_i)/log(min_j V_t_j)`` so larger-vocab teachers don't
            dominate purely by vocab size.
        alpha: Softmax temperature on the dynamic teacher-weight scores
            (``sum_weights_metric``). Inert when weights are static.
        sum_weights_metric: sum-mode only — ``"ce"`` / ``"entropy"`` /
            ``"max_prob"`` drives dynamic per-teacher weights; ``None`` (default)
            uses the static ``teacher_weights``.
        student_vocab_size: Full student tokenizer vocab size (sizes the
            projection matrices' V_s axis). Runtime-injected from
            ``len(student_tokenizer)``; not a user YAML knob.
        teacher_vocab_sizes / projection_matrix_paths / teacher_weights /
            teacher_gold_loss / teacher_xtoken_loss: Parallel per-teacher lists
            (one entry per ``teachers[i]``), runtime-injected by the driver from
            the ``teachers`` config. A ``None`` projection path marks a
            same-tokenizer teacher (direct KL, no projection / alignment). Not
            user loss_fn keys in YAML.
    """

    gold_loss: bool
    xtoken_loss: bool
    temperature: float
    vocab_topk: int
    uncommon_topk: int
    reverse_kl: bool
    exact_token_match_only: bool
    kl_loss_weight: float
    ce_loss_scale: float
    dynamic_loss_scaling: bool
    # Multi-teacher aggregation (user loss_fn knobs).
    kd_loss_mode: str  # "sum" | "averaged_logits" | "select_teacher"
    normalize_teacher_by_vocab: bool  # sum-mode only
    alpha: float  # softmax temperature on dynamic teacher-weight scores
    sum_weights_metric: NotRequired[
        Optional[str]
    ]  # "ce" | "entropy" | "max_prob"; None => static teacher_weights. sum-mode only.
    # Runtime-injected parallel per-teacher lists (+ the student vocab size); not
    # user loss_fn keys in YAML.
    student_vocab_size: NotRequired[int]
    teacher_vocab_sizes: NotRequired[list[int]]
    projection_matrix_paths: NotRequired[list[Optional[str]]]
    teacher_weights: NotRequired[list[float]]
    teacher_gold_loss: NotRequired[list[Optional[bool]]]
    teacher_xtoken_loss: NotRequired[list[Optional[bool]]]


class CrossTokenizerDistillationLossDataDict(TypedDict):
    """Student-side keys are fixed; teacher-side keys are teacher-indexed.

    Only the student keys below are static. Each teacher ``i`` contributes a
    dynamic set produced by ``CrossTokenizerCollator`` / the transport and so
    cannot be enumerated here:

    - Every teacher: ``teacher_{i}_full_logits_ipc`` (node-local CUDA IPC) or
      ``teacher_{i}_full_logits_cross_cluster`` (data_plane) — rebuilt to
      full-vocab teacher logits by the loss-input keystone.
    - Cross-tokenizer teacher only: ``teacher_{i}_input_ids`` /
      ``teacher_{i}_token_mask`` ``[B, T_t]`` and ``alignment_{i}_*``
      (pair_valid / pair_is_correct / chunk ids / partition masks / num_chunks).
    - Same-tokenizer teacher (``projection_matrix_paths[i] is None``): no
      ``teacher_{i}_input_ids`` / ``alignment_{i}_*``; reuses the student
      tokenization (identity 1:1 alignment).
    """

    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    token_mask: torch.Tensor
    sample_mask: torch.Tensor


class CrossTokenizerDistillationLossFn(LossFunction):
    """Multi-teacher cross-tokenizer distillation loss.

    Each teacher contributes a KD term; the per-teacher terms are aggregated per
    ``kd_loss_mode`` and combined with a single student next-token CE. The
    single-teacher path is just ``num_teachers == 1``.

    Per-teacher KD path is selected by that teacher's tokenizer kind and its
    ``(gold_loss, xtoken_loss)`` flags:

    - Cross-tokenizer teacher (``projection_matrix_paths[i]`` set):
      - ``(False, False)`` -> P-KL: full-vocab projection KL (student logits
        mapped through the projection matrix M) over a microbatch-global top-k
        teacher subset.
      - ``(True, False)`` -> gold-loss: KL on the exact-mapped *common* partition
        plus a sorted-L1 term on the *uncommon* tail.
      - ``(True, True)`` -> gold-loss with the xtoken modifier: relaxed exact-map
        threshold (``>= 0.6``) + collision replacement.
    - Same-tokenizer teacher (``projection_matrix_paths[i] is None``): direct
      top-k per-position KL, no projection / no alignment.

    ``(False, True)`` is rejected in ``__init__``.

    Aggregation (``kd_loss_mode``): ``sum`` (weighted sum; static
    ``teacher_weights`` or dynamic ``sum_weights_metric``), ``averaged_logits``
    (convex-average teacher logits then one KL; same-tokenizer teachers only),
    ``select_teacher`` (use only the lowest-CE teacher).

    Inputs (via ``LossInputType.DISTILLATION_CROSS_TOKENIZER``): ``logits`` (raw
    student logits), ``student_logits_contig`` (CP-relaid), the per-teacher
    ``teacher_full_logits_by_idx`` (rebuilt by the loss-input keystone) and
    ``aligns_by_idx`` (localized + next-token-shifted per teacher), and the TP/CP
    groups.
    """

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.DISTILLATION_CROSS_TOKENIZER

    def __init__(self, cfg: CrossTokenizerDistillationLossConfig):
        if cfg["xtoken_loss"] and not cfg["gold_loss"]:
            raise ValueError(
                "xtoken_loss=True requires gold_loss=True; xtoken_loss is a "
                "modifier inside the gold path (relaxes the exact-map threshold "
                "and adds collision resolution) and is undefined in the P-KL path."
            )
        # Dynamic teacher weighting (sum_weights_metric) and
        # normalize_teacher_by_vocab only apply in kd_loss_mode="sum"; reject the
        # combo rather than silently ignore it under the other modes.
        _sum_weights_metric = cfg.get("sum_weights_metric")
        if _sum_weights_metric is not None and cfg["kd_loss_mode"] != "sum":
            raise ValueError(
                f"sum_weights_metric={_sum_weights_metric!r} is only applied "
                f"in kd_loss_mode='sum'; it is ignored by '{cfg['kd_loss_mode']}'. "
                "Unset one of them."
            )
        if cfg.get("normalize_teacher_by_vocab") and cfg["kd_loss_mode"] != "sum":
            raise ValueError(
                "normalize_teacher_by_vocab is only applied in kd_loss_mode='sum'; "
                f"it is ignored by '{cfg['kd_loss_mode']}'. Unset one of them."
            )
        # averaged_logits forms a convex combination of teacher logits
        # (weight_i / sum(weights)); a zero weight-sum makes that division
        # undefined. Reject here rather than fail with a deep error mid-step.
        _weights = cfg.get("teacher_weights")
        if (
            cfg["kd_loss_mode"] == "averaged_logits"
            and _weights is not None
            and sum(_weights) == 0
        ):
            raise ValueError(
                "teacher_weights must not sum to zero in "
                "kd_loss_mode='averaged_logits' (they form a convex combination "
                f"of teacher logits); got teacher_weights={list(_weights)}."
            )
        # Global loss knobs (shared across all teachers).
        self.gold_loss = cfg["gold_loss"]
        self.xtoken_loss = cfg["xtoken_loss"]
        self.temperature = cfg["temperature"]
        self.vocab_topk = cfg["vocab_topk"]
        self.uncommon_topk = cfg["uncommon_topk"]
        self.reverse_kl = cfg["reverse_kl"]
        self.exact_token_match_only = cfg["exact_token_match_only"]
        self.kl_loss_weight = cfg["kl_loss_weight"]
        self.ce_loss_scale = cfg["ce_loss_scale"]
        self.dynamic_loss_scaling = cfg["dynamic_loss_scaling"]
        # Multi-teacher aggregation knobs.
        self.kd_loss_mode = cfg["kd_loss_mode"]
        self.normalize_teacher_by_vocab = cfg["normalize_teacher_by_vocab"]
        self.alpha = cfg["alpha"]
        # sum_weights_metric is NotRequired -> None means static teacher_weights.
        self.sum_weights_metric = cfg.get("sum_weights_metric")
        # student_vocab_size + the per-teacher lists are runtime-injected by the
        # driver from the tokenizer lengths / teachers config (not user YAML
        # knobs), so they're NotRequired in the schema but must be present when
        # the loss is constructed.
        student_vocab_size = cfg.get("student_vocab_size")
        assert student_vocab_size is not None, (
            "student_vocab_size must be injected into the cross-tokenizer loss "
            "config by the driver before the loss is built."
        )
        self.student_vocab_size = student_vocab_size
        projection_matrix_paths = cfg.get("projection_matrix_paths")
        teacher_vocab_sizes = cfg.get("teacher_vocab_sizes")
        teacher_weights = cfg.get("teacher_weights")
        teacher_gold_loss = cfg.get("teacher_gold_loss")
        teacher_xtoken_loss = cfg.get("teacher_xtoken_loss")
        assert (
            projection_matrix_paths is not None
            and teacher_vocab_sizes is not None
            and teacher_weights is not None
            and teacher_gold_loss is not None
            and teacher_xtoken_loss is not None
        ), (
            "projection_matrix_paths / teacher_vocab_sizes / teacher_weights / "
            "teacher_gold_loss / teacher_xtoken_loss must be injected into the "
            "cross-tokenizer loss config by the driver (one entry per teacher) "
            "before the loss is built."
        )
        self.projection_matrix_paths = list(projection_matrix_paths)
        self.teacher_vocab_sizes = list(teacher_vocab_sizes)
        self.teacher_weights = list(teacher_weights)
        self.teacher_gold_loss = list(teacher_gold_loss)
        self.teacher_xtoken_loss = list(teacher_xtoken_loss)
        # Every per-teacher list must have the same length (one entry per
        # teacher); a mismatch would otherwise surface as a deep IndexError
        # mid-training instead of a clear error here.
        per_teacher_lens = {
            "projection_matrix_paths": len(self.projection_matrix_paths),
            "teacher_vocab_sizes": len(self.teacher_vocab_sizes),
            "teacher_weights": len(self.teacher_weights),
            "teacher_gold_loss": len(self.teacher_gold_loss),
            "teacher_xtoken_loss": len(self.teacher_xtoken_loss),
        }
        if len(set(per_teacher_lens.values())) != 1:
            raise ValueError(
                f"per-teacher lists must be equal length, got {per_teacher_lens}"
            )
        self.num_teachers = len(self.projection_matrix_paths)
        # The materialized projection matrices and the derived exact-map
        # partitions live in process-local caches in x_token.loss_utils (keyed by
        # path), not on this instance — keeps the driver-side loss_fn free of
        # large CUDA tensors and lets multiple teachers / loss instances share one
        # load per path.

    def _teacher_is_same_vocab(self, i: int) -> bool:
        """A teacher is same-vocab (direct KL, no projection) iff its path is None."""
        return self.projection_matrix_paths[i] is None

    def __call__(  # type: ignore[override]
        self,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        logits: torch.Tensor,
        student_logits_contig: torch.Tensor,
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        *,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the (multi-teacher) cross-tokenizer distillation loss.

        Per-teacher KD terms are aggregated per ``kd_loss_mode`` and combined with
        a single student next-token CE — dynamic-scaled when
        ``dynamic_loss_scaling`` is set (KD rescaled to the detached CE magnitude,
        ``kl_loss_weight`` / ``ce_loss_scale`` ignored), else fixed-weighted. The
        single-teacher path is just ``num_teachers == 1``.

        ``student_logits_contig`` (CP-relaid) and the per-teacher ``aligns_by_idx``
        / ``teacher_full_logits_by_idx`` are precomputed in the loss-input
        keystone; the raw ``logits`` is kept for the CE term.
        """
        ce_loss = self._compute_ce(logits, data, global_valid_toks)

        if self.kd_loss_mode == "sum":
            total_kd, per_teacher_metrics = self._sum_kd(
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                global_valid_toks,
                tp_group=tp_group,
                cp_group=cp_group,
            )
        elif self.kd_loss_mode == "averaged_logits":
            total_kd, per_teacher_metrics = self._averaged_logits_kd(
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                global_valid_toks,
                tp_group=tp_group,
                cp_group=cp_group,
            )
        elif self.kd_loss_mode == "select_teacher":
            total_kd, per_teacher_metrics = self._select_teacher_kd(
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                global_valid_toks,
                tp_group=tp_group,
                cp_group=cp_group,
            )
        else:
            raise ValueError(f"Unknown kd_loss_mode: {self.kd_loss_mode!r}")

        # Combine the aggregated KD term with the single student CE term.
        if self.dynamic_loss_scaling:
            # loss = sg(ce/kd) * kd + ce; user kl_loss_weight / ce_loss_scale
            # are intentionally ignored in this branch.
            kd_detached = total_kd.detach().abs()
            ce_detached = ce_loss.detach().abs()
            kl_scale = torch.where(
                kd_detached > 0,
                ce_detached / kd_detached,
                torch.ones_like(kd_detached),
            )
            loss = kl_scale * total_kd + ce_loss
        else:
            kl_scale = torch.tensor(1.0, device=total_kd.device, dtype=total_kd.dtype)
            loss = self.kl_loss_weight * total_kd + self.ce_loss_scale * ce_loss

        # Next-token accuracy on the student side (quick per-step signal). Computed
        # once from the shared CP-relaid fields carried on every teacher's align;
        # the CP-aware shift pairs predictors with the right labels.
        align0 = aligns_by_idx[0]
        assert (
            align0.student_input_ids is not None
            and align0.student_token_mask is not None
        ), "the keystone sets the CP-relaid student fields on every teacher's align"
        accuracy = next_token_accuracy(
            student_logits_contig,
            input_ids=align0.student_input_ids,
            token_mask=align0.student_token_mask,
            sample_mask=data["sample_mask"],
            tp_group=tp_group,
            cp_group=cp_group,
        )

        metrics: dict[str, Any] = {
            "loss": loss.item(),
            # Aggregate KD term (kept under ``kl_loss`` so existing trainer metric
            # handling keeps working); per-teacher terms are suffixed ``_t{i}``.
            "kl_loss": total_kd.item(),
            "ce_loss": ce_loss.item(),
            "kl_loss_scale": kl_scale.item(),
            "accuracy": accuracy.item(),
            "num_valid_samples": data["input_ids"].shape[0],
        }
        metrics.update(per_teacher_metrics)
        return loss, metrics

    # Multi-teacher aggregation
    def _resolve_gold_xtoken(self, i: int, use_per_teacher: bool) -> tuple[bool, bool]:
        """Effective ``(gold_loss, xtoken_loss)`` for teacher ``i``.

        Per-teacher overrides are honored only when ``use_per_teacher`` is set
        (``sum`` mode); ``select_teacher`` / ``averaged_logits`` use the global
        flags. A ``None`` override falls back to the global value.
        """
        if not use_per_teacher:
            return self.gold_loss, self.xtoken_loss
        g = self.teacher_gold_loss[i]
        x = self.teacher_xtoken_loss[i]
        gold = g if g is not None else self.gold_loss
        xtoken = x if x is not None else self.xtoken_loss
        return gold, xtoken

    def _compute_teacher_kd(
        self,
        i: int,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        use_per_teacher_flags: bool,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """KD term for teacher ``i`` plus its (unsuffixed) metrics.

        Dispatches on tokenizer kind: same-vocab -> direct top-k per-position KL;
        cross-tokenizer -> P-KL or gold path using teacher ``i``'s projection and
        its localized alignment. Both consume the shared CP-relaid student logits.
        """
        if self._teacher_is_same_vocab(i):
            return self._compute_same_vocab_kl(
                student_logits_contig,
                teacher_full_logits_by_idx[i],
                aligns_by_idx[i],
                global_valid_toks,
                tp_group=tp_group,
                cp_group=cp_group,
            )

        teacher_logits = teacher_full_logits_by_idx[i]
        align = aligns_by_idx[i]
        proj_path = self.projection_matrix_paths[i]
        v_t = self.teacher_vocab_sizes[i]
        gold, xtoken = self._resolve_gold_xtoken(i, use_per_teacher_flags)
        if xtoken and not gold:
            raise ValueError(f"teacher {i}: xtoken_loss=True requires gold_loss=True.")

        if gold:
            kd, kl_common, l1_uncommon, num_valid_chunks, top1 = self._compute_gold(
                student_logits_contig,
                teacher_logits,
                align,
                projection_matrix_path=proj_path,
                teacher_vocab_size=v_t,
                xtoken_loss=xtoken,
                tp_group=tp_group,
                cp_group=cp_group,
            )
            return kd, {
                "kl_loss": kd.item(),
                "kl_common": kl_common.item(),
                "l1_uncommon": l1_uncommon.item(),
                "proj_accuracy": top1.item(),
                "num_valid_chunks": int(num_valid_chunks.item()),
            }

        kl, num_valid_pairs, proj_acc = self._compute_p_kl(
            student_logits_contig,
            teacher_logits,
            align,
            projection_matrix_path=proj_path,
            teacher_vocab_size=v_t,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        return kl, {
            "kl_loss": kl.item(),
            "proj_accuracy": proj_acc.item(),
            "num_valid_pairs": int(num_valid_pairs.item()),
        }

    def _compute_same_vocab_kl(
        self,
        student_logits_contig: torch.Tensor,
        teacher_full_logits: torch.Tensor,
        align: LocalizedAlignment,
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Direct top-k per-position KL for a same-tokenizer teacher.

        Identical tokenizer => teacher tokens == student tokens (identity position
        alignment), so no projection / no chunk-averaging. The reduction matches
        CE: masked next-token mean normalized by ``global_valid_toks``, scaled by
        ``T**2``.
        """
        kd = self._direct_topk_kl(
            student_logits_contig,
            teacher_full_logits,
            align,
            global_valid_toks,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        return kd, {"kl_loss": kd.item()}

    def _direct_topk_kl(
        self,
        student_logits: torch.Tensor,
        teacher_full_logits: torch.Tensor,
        align: LocalizedAlignment,
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """Top-K per-position KL on a shared vocab (same tokenizer), TP/CP-aware.

        Top-K columns are selected at the student from the reassembled full-vocab
        teacher logits (``select_teacher_topk_indices`` MAX-reduces across CP so
        every CP rank agrees on the columns). The student is gathered to full vocab
        across TP (``vocab_parallel_full_log_softmax``) *before* slicing — slicing
        a TP-local shard would pick the wrong columns. Both sides are renormalized
        within the K-subset (mathematically identical; the full-vocab partition
        function cancels). The masked next-token mean is normalized by the
        CP/DP-global valid-token count, exactly like the CE term.
        """
        T = self.temperature
        # Drop HF lm_head padding beyond the shared tokenizer vocab.
        v_s = self.student_vocab_size
        teacher = teacher_full_logits
        if teacher.shape[-1] > v_s:
            teacher = teacher[..., :v_s]
        vocab_topk = min(self.vocab_topk, teacher.shape[-1])
        topk_idx = select_teacher_topk_indices(teacher, vocab_topk, cp_group=cp_group)

        student_log_probs = vocab_parallel_full_log_softmax(
            student_logits, T, tp_group=tp_group
        )
        student_gathered = student_log_probs[..., topk_idx]
        student_log_probs_k = student_gathered - torch.logsumexp(
            student_gathered, dim=-1, keepdim=True
        )
        teacher_log_probs_k = torch.log_softmax(
            teacher[..., topk_idx].float() / T, dim=-1
        )
        if self.reverse_kl:
            per_pos = torch.nn.functional.kl_div(
                teacher_log_probs_k,
                student_log_probs_k,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
        else:
            per_pos = torch.nn.functional.kl_div(
                student_log_probs_k,
                teacher_log_probs_k,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
        return self._same_vocab_masked_kl(per_pos, align, global_valid_toks, cp_group)

    def _direct_full_vocab_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        align: LocalizedAlignment,
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """Full-vocab per-position KL on a shared vocab (same tokenizer), TP/CP-aware.

        Used by ``averaged_logits`` over the convex-averaged teacher logits. The
        student is gathered to full vocab across TP; the teacher (already full
        vocab) is sliced to the student width to drop HF lm_head padding.
        """
        T = self.temperature
        student_log_probs = vocab_parallel_full_log_softmax(
            student_logits, T, tp_group=tp_group
        )
        v_s = student_log_probs.shape[-1]
        teacher = teacher_logits.float()
        if teacher.shape[-1] > v_s:
            teacher = teacher[..., :v_s]
        teacher_log_probs = torch.log_softmax(teacher / T, dim=-1)
        if self.reverse_kl:
            per_pos = torch.nn.functional.kl_div(
                teacher_log_probs, student_log_probs, reduction="none", log_target=True
            ).sum(dim=-1)
        else:
            per_pos = torch.nn.functional.kl_div(
                student_log_probs, teacher_log_probs, reduction="none", log_target=True
            ).sum(dim=-1)
        return self._same_vocab_masked_kl(per_pos, align, global_valid_toks, cp_group)

    def _same_vocab_masked_kl(
        self,
        per_pos: torch.Tensor,
        align: LocalizedAlignment,
        global_valid_toks: torch.Tensor,
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """Masked next-token mean of a per-position KL (same-tokenizer reduction).

        ``per_pos`` is the per-position KL on this CP rank's contiguous window
        (student and teacher position-aligned). The CP-aware next-token shift
        (:func:`cp_shift_next`) selects positions whose target token (p+1) is a
        valid label — at CP=1 this is the plain ``token_mask[:, 1:]`` shift.
        Reduction matches CE: ``masked_mean`` over ``global_valid_toks``, scaled by
        ``T**2``.
        """
        T = self.temperature
        assert align.student_token_mask is not None, (
            "same-tokenizer KD reads the keystone-set student token mask"
        )
        next_mask = cp_shift_next(
            to_local_if_dtensor(align.student_token_mask), cp_group, fill=0
        )
        sample_mask = to_local_if_dtensor(align.sample_mask)
        mask = next_mask.float() * sample_mask.unsqueeze(-1).float()
        return (
            masked_mean(per_pos, mask, global_normalization_factor=global_valid_toks)
            * T
            * T
        )

    def _sum_kd(
        self,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Weighted sum: ``total_kd = Σ_i weight_i · KD_i``.

        Weights are static (config ``weight``) or dynamic (``sum_weights_metric``).
        When ``normalize_teacher_by_vocab`` is set, each teacher's KD is
        additionally scaled by ``log(V_t_i) / log(min_j V_t_j)``.
        """
        device = student_logits_contig.device
        if self.sum_weights_metric is not None:
            weights = self._compute_dynamic_weights(
                data, teacher_full_logits_by_idx, aligns_by_idx
            )
        else:
            weights = [
                torch.tensor(
                    self.teacher_weights[i],
                    device=device,
                    dtype=student_logits_contig.dtype,
                )
                for i in range(self.num_teachers)
            ]

        # Bound unconditionally (cheap) so the per-teacher scale below is never
        # read unbound; only consumed when normalize_teacher_by_vocab is set.
        temp_weight = torch.log(
            torch.tensor(float(min(self.teacher_vocab_sizes)), device=device)
        )

        total_kd: Optional[torch.Tensor] = None
        per_metrics: dict[str, Any] = {}
        # Deterministic teacher order: each teacher's KD fires its own collectives,
        # so the order must match across ranks.
        for i in range(self.num_teachers):
            kd_i, m_i = self._compute_teacher_kd(
                i,
                student_logits_contig,
                data,
                teacher_full_logits_by_idx,
                aligns_by_idx,
                global_valid_toks,
                use_per_teacher_flags=True,
                tp_group=tp_group,
                cp_group=cp_group,
            )
            weighted = kd_i * weights[i]
            if self.normalize_teacher_by_vocab:
                v_scale = (
                    torch.log(
                        torch.tensor(float(self.teacher_vocab_sizes[i]), device=device)
                    )
                    / temp_weight
                )
                weighted = weighted * v_scale
            total_kd = weighted if total_kd is None else total_kd + weighted
            for k, v in m_i.items():
                per_metrics[f"{k}_t{i}"] = v
            per_metrics[f"weight_t{i}"] = float(weights[i].item())
        assert total_kd is not None
        return total_kd, per_metrics

    def _averaged_logits_kd(
        self,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Convex-weighted average of teacher logits, then one direct KL.

        Valid only when all teachers are same-tokenizer (no projection) and ship
        full logits of identical shape. Otherwise falls back to a plain
        static-weight sum (no dynamic weights, no ``normalize_teacher_by_vocab``).
        """
        full = [teacher_full_logits_by_idx.get(i) for i in range(self.num_teachers)]
        # Direct per-position KL is only valid when every teacher shares the
        # student's tokenizer (no projection) *and* ships full logits of identical
        # shape. Two cross-tokenizer teachers can have matching shapes yet still
        # need the projection/alignment path, so the shape check alone is
        # insufficient.
        same_tokenizer = all(p is None for p in self.projection_matrix_paths)
        same_shape = all(f is not None for f in full) and (
            len({tuple(f.shape) for f in full if f is not None}) == 1
        )
        if not (same_tokenizer and same_shape):
            total_kd: Optional[torch.Tensor] = None
            per_metrics: dict[str, Any] = {}
            for i in range(self.num_teachers):
                kd_i, m_i = self._compute_teacher_kd(
                    i,
                    student_logits_contig,
                    data,
                    teacher_full_logits_by_idx,
                    aligns_by_idx,
                    global_valid_toks,
                    use_per_teacher_flags=False,
                    tp_group=tp_group,
                    cp_group=cp_group,
                )
                w = self.teacher_weights[i]
                weighted = kd_i * w
                total_kd = weighted if total_kd is None else total_kd + weighted
                for k, v in m_i.items():
                    per_metrics[f"{k}_t{i}"] = v
                per_metrics[f"weight_t{i}"] = float(w)
            assert total_kd is not None
            return total_kd, per_metrics

        total_w = sum(self.teacher_weights)
        avg: Optional[torch.Tensor] = None
        for i, f in enumerate(full):
            assert f is not None
            contrib = f.float() * (self.teacher_weights[i] / total_w)
            avg = contrib if avg is None else avg + contrib
        assert avg is not None
        kd = self._direct_full_vocab_kl(
            student_logits_contig,
            avg,
            aligns_by_idx[0],
            global_valid_toks,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        return kd, {"kl_loss": kd.item()}

    def _dp_global_masked_mean(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked mean of ``values`` over the *process-global* valid count.

        Teacher selection / dynamic weighting must be identical on every rank: a
        rank-local mean lets ranks pick a different teacher / different weights,
        and the per-teacher KD's collectives then see divergent participation
        (deadlock when one rank's choice fires a collective another's does not).
        All-reduce the masked sum and the mask count over the full group so every
        rank gets the same score. The result is detached (it gates selection /
        weighting and is not back-propagated).
        """
        num = group_all_reduce_sum(
            (values * mask).sum(), group=torch.distributed.group.WORLD
        )
        den = group_all_reduce_sum(
            mask.sum(), group=torch.distributed.group.WORLD
        ).clamp(min=1.0)
        return num / den

    def _select_teacher_kd(
        self,
        student_logits_contig: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
        global_valid_toks: torch.Tensor,
        *,
        tp_group: Optional[torch.distributed.ProcessGroup],
        cp_group: Optional[torch.distributed.ProcessGroup],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Use only the teacher with the lowest next-token CE on its own tokens."""
        with torch.no_grad():
            ces: list[float] = []
            for i in range(self.num_teachers):
                t_logits, t_ids, t_mask = self._teacher_score_inputs(
                    i, data, teacher_full_logits_by_idx, aligns_by_idx
                )
                ce_pos = torch.nn.functional.cross_entropy(
                    t_logits[:, :-1].reshape(-1, t_logits.shape[-1]).float(),
                    t_ids[:, 1:].reshape(-1),
                    reduction="none",
                )
                mask = (
                    t_mask[:, 1:].float() * data["sample_mask"].unsqueeze(-1).float()
                ).reshape(-1)
                ces.append(self._dp_global_masked_mean(ce_pos, mask).item())
            best = int(min(range(self.num_teachers), key=lambda j: ces[j]))

        kd, m = self._compute_teacher_kd(
            best,
            student_logits_contig,
            data,
            teacher_full_logits_by_idx,
            aligns_by_idx,
            global_valid_toks,
            use_per_teacher_flags=False,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        per_metrics: dict[str, Any] = {f"{k}_t{best}": v for k, v in m.items()}
        per_metrics["selected_teacher"] = best
        return kd, per_metrics

    def _teacher_score_inputs(
        self,
        i: int,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logits, input_ids, token_mask)`` for teacher ``i``'s scores.

        The token mask is over the tokenization the score is computed on: the
        shared student tokens (CP-relaid) for a same-vocab teacher, teacher ``i``'s
        own otherwise. Every teacher ships full logits, so the full distribution is
        always available.
        """
        if self._teacher_is_same_vocab(i):
            align = aligns_by_idx[i]
            assert (
                align.student_input_ids is not None
                and align.student_token_mask is not None
            ), "the keystone sets the CP-relaid student fields on every align"
            ids = to_local_if_dtensor(align.student_input_ids)
            token_mask = to_local_if_dtensor(align.student_token_mask)
        else:
            ids = to_local_if_dtensor(data[f"teacher_{i}_input_ids"])
            token_mask = to_local_if_dtensor(data[f"teacher_{i}_token_mask"])
        return teacher_full_logits_by_idx[i], ids, token_mask

    def _compute_dynamic_weights(
        self,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        teacher_full_logits_by_idx: dict[int, torch.Tensor],
        aligns_by_idx: dict[int, LocalizedAlignment],
    ) -> list[torch.Tensor]:
        """Sequence-level dynamic teacher weights via ``sum_weights_metric``.

        Per teacher computes a scalar score (``ce`` -> -CE, ``entropy`` ->
        -entropy, ``max_prob`` -> max prob; higher = more trusted), optionally
        rescaled by ``log(V_t_i)/log(min_j V_t_j)``, then ``softmax(alpha *
        scores)`` across teachers.
        """
        device = data["input_ids"].device
        # Bound unconditionally (cheap); only consumed when
        # normalize_teacher_by_vocab is set.
        temp_weight = torch.log(
            torch.tensor(float(min(self.teacher_vocab_sizes)), device=device)
        )
        scores: list[torch.Tensor] = []
        for i in range(self.num_teachers):
            t_logits, t_ids, t_mask = self._teacher_score_inputs(
                i, data, teacher_full_logits_by_idx, aligns_by_idx
            )
            score = self._teacher_weight_score(
                t_logits, t_ids, t_mask, data["sample_mask"]
            )
            if self.normalize_teacher_by_vocab:
                v_log = torch.log(
                    torch.tensor(float(self.teacher_vocab_sizes[i]), device=device)
                )
                score = score * (v_log / temp_weight)
            scores.append(score)
        weights = torch.softmax(self.alpha * torch.stack(scores), dim=0)
        return [weights[i] for i in range(self.num_teachers)]

    def _teacher_weight_score(
        self,
        t_logits: torch.Tensor,
        t_ids: torch.Tensor,
        t_mask: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar weight-metric score for one teacher (higher = more trusted).

        Padded positions and masked-out samples are excluded, so long-padded
        batches don't let near-uniform padding logits dominate the score.
        """
        samp = to_local_if_dtensor(sample_mask).unsqueeze(-1).float()
        if self.sum_weights_metric == "ce":
            ce_pos = torch.nn.functional.cross_entropy(
                t_logits[:, :-1].reshape(-1, t_logits.shape[-1]).float(),
                t_ids[:, 1:].reshape(-1),
                reduction="none",
            )
            mask = (t_mask[:, 1:].float() * samp).reshape(-1)
            return -self._dp_global_masked_mean(ce_pos, mask)
        mask = t_mask.float() * samp
        if self.sum_weights_metric == "entropy":
            probs = torch.softmax(t_logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            return -self._dp_global_masked_mean(entropy, mask)
        if self.sum_weights_metric == "max_prob":
            probs = torch.softmax(t_logits.float(), dim=-1)
            return self._dp_global_masked_mean(probs.max(dim=-1).values, mask)
        raise ValueError(f"Unknown sum_weights_metric: {self.sum_weights_metric!r}")

    def _compute_p_kl(
        self,
        student_logits: torch.Tensor,
        teacher_full_logits: torch.Tensor,
        align: LocalizedAlignment,
        *,
        projection_matrix_path: Optional[str],
        teacher_vocab_size: int,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """P-KL: chunk-averaged KL over a microbatch-global top-k teacher subset.

        ``projection_matrix_path`` / ``teacher_vocab_size`` are teacher ``i``'s
        (the student vocab size is shared on ``self``).
        """
        # This path only runs for a cross-tokenizer teacher, so its projection and
        # chunk-alignment fields are populated (narrows the Optional fields).
        assert projection_matrix_path is not None, (
            "P-KL requires a projection matrix (cross-tokenizer teacher)"
        )
        assert (
            align.student_chunk_id is not None
            and align.teacher_chunk_id is not None
            and align.pair_valid is not None
            and align.pair_is_correct is not None
        ), "the keystone populates the chunk alignment for a cross-tokenizer teacher"
        T = self.temperature
        device = student_logits.device
        eps = 1e-10

        student_log_probs = vocab_parallel_log_softmax(
            student_logits, T, tp_group=tp_group
        )
        student_probs = student_log_probs.exp()

        sparse_projection = get_sparse_projection_matrix(
            projection_matrix_path,
            device,
            student_vocab_size=self.student_vocab_size,
            teacher_vocab_size=teacher_vocab_size,
        )
        projected_full = project_student_to_teacher_vocab(
            student_probs, sparse_projection, tp_group=tp_group
        )
        full_teacher_vocab_size = projected_full.shape[-1]

        # HF models commonly pad lm_head out_features beyond len(tokenizer); the
        # projection is sized to the real tokenizer vocab, so slice the teacher
        # logits to the same V_t.
        if teacher_full_logits.shape[-1] > full_teacher_vocab_size:
            teacher_full_logits = teacher_full_logits[..., :full_teacher_vocab_size]

        student_chunk_id = align.student_chunk_id
        teacher_chunk_id = align.teacher_chunk_id
        pair_valid = align.pair_valid
        if self.exact_token_match_only:
            pair_valid = pair_valid & align.pair_is_correct
        max_chunks = pair_valid.shape[1]

        vocab_topk = min(self.vocab_topk, full_teacher_vocab_size)
        global_top_indices = select_teacher_topk_indices(
            teacher_full_logits, vocab_topk, cp_group=cp_group
        )

        projected_topk = projected_full[..., global_top_indices]
        teacher_topk_logits = teacher_full_logits[..., global_top_indices]
        target_log_probs = torch.log_softmax(teacher_topk_logits / T, dim=-1)

        proj_chunks, proj_sizes = chunk_average_log_probs(
            projected_topk, student_chunk_id, max_chunks, cp_group=cp_group
        )
        tgt_log_chunks, tgt_sizes = chunk_average_log_probs(
            target_log_probs, teacher_chunk_id, max_chunks, cp_group=cp_group
        )

        proj_chunks = proj_chunks / (proj_chunks.sum(dim=-1, keepdim=True) + eps)
        proj_log_chunks = (proj_chunks + eps).log()

        chunk_mask = valid_chunk_mask(proj_sizes, tgt_sizes, pair_valid)
        sample_mask_bool = align.sample_mask.bool()
        valid_bool = chunk_mask & sample_mask_bool.unsqueeze(-1)
        global_valid_chunks = group_all_reduce_sum(
            valid_bool.sum().to(torch.float32), group=torch.distributed.group.WORLD
        )
        if global_valid_chunks.item() == 0:
            zero = torch.zeros((), device=device, dtype=proj_log_chunks.dtype)
            return (
                zero,
                torch.zeros((), device=device, dtype=torch.long),
                zero.detach(),
            )

        with torch.no_grad():
            proj_top1 = proj_chunks.argmax(dim=-1)
            tgt_top1 = torch.exp(tgt_log_chunks).argmax(dim=-1)
            proj_matches = (proj_top1 == tgt_top1) & chunk_mask
            proj_acc = proj_matches.sum().float() / chunk_mask.sum().float().clamp(
                min=1.0
            )

        if self.reverse_kl:
            per_chunk_kl = torch.nn.functional.kl_div(
                tgt_log_chunks, proj_log_chunks, reduction="none", log_target=True
            ).sum(dim=-1)
        else:
            per_chunk_kl = torch.nn.functional.kl_div(
                proj_log_chunks, tgt_log_chunks, reduction="none", log_target=True
            ).sum(dim=-1)

        sample_mask = align.sample_mask.to(per_chunk_kl.dtype)
        valid = chunk_mask.to(per_chunk_kl.dtype) * sample_mask.unsqueeze(-1)
        denom = global_valid_chunks.to(per_chunk_kl.dtype).clamp(min=1.0)
        kl_loss = (per_chunk_kl * valid).sum() / denom * (T * T)

        return kl_loss, valid.sum().detach(), proj_acc.detach()

    def _compute_gold(
        self,
        student_logits: torch.Tensor,
        teacher_full_logits: torch.Tensor,
        align: LocalizedAlignment,
        *,
        projection_matrix_path: Optional[str],
        teacher_vocab_size: int,
        xtoken_loss: bool,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gold-loss: KL on the common (exact-mapped) vocab + L1 on the uncommon tail.

        ``projection_matrix_path`` / ``teacher_vocab_size`` / ``xtoken_loss`` are
        teacher ``i``'s (resolved against the global defaults by the caller).
        """
        # Gold only runs for a cross-tokenizer teacher, so its projection and
        # chunk-alignment fields are populated (narrows the Optional fields).
        assert projection_matrix_path is not None, (
            "gold-loss requires a projection matrix (cross-tokenizer teacher)"
        )
        assert (
            align.student_chunk_id is not None
            and align.teacher_chunk_id is not None
            and align.pair_valid is not None
        ), "the keystone populates the chunk alignment for a cross-tokenizer teacher"
        T = self.temperature
        device = student_logits.device

        exact_map = build_exact_token_map(
            projection_matrix_path,
            device,
            xtoken_loss=xtoken_loss,
            teacher_vocab_size=teacher_vocab_size,
        )
        common_s = exact_map["common_student"]
        common_t = exact_map["common_teacher"]
        uncommon_s = exact_map["uncommon_student"]
        uncommon_t = exact_map["uncommon_teacher"]
        v_teacher = teacher_vocab_size

        if teacher_full_logits.shape[-1] > v_teacher:
            teacher_full_logits = teacher_full_logits[..., :v_teacher]

        # common_s / uncommon_s are arbitrary V_s indices, so the gold path needs
        # full-vocab student log-probs (TP-sharded students are all-gathered).
        student_log_probs = vocab_parallel_full_log_softmax(
            student_logits, T, tp_group=tp_group
        )
        teacher_log_probs = torch.log_softmax(teacher_full_logits / T, dim=-1)

        student_chunk_id = align.student_chunk_id
        teacher_chunk_id = align.teacher_chunk_id
        pair_valid = align.pair_valid
        max_chunks = pair_valid.shape[1]

        student_chunks, s_sizes = chunk_average_log_probs(
            student_log_probs, student_chunk_id, max_chunks, cp_group=cp_group
        )
        teacher_chunks, t_sizes = chunk_average_log_probs(
            teacher_log_probs, teacher_chunk_id, max_chunks, cp_group=cp_group
        )

        chunk_mask = valid_chunk_mask(s_sizes, t_sizes, pair_valid)
        sample_mask = align.sample_mask
        valid_chunk = chunk_mask & sample_mask.bool().unsqueeze(-1)
        zero_dtype = student_log_probs.dtype
        global_valid_chunks = group_all_reduce_sum(
            valid_chunk.sum().to(torch.float32), group=torch.distributed.group.WORLD
        )
        if global_valid_chunks.item() == 0:
            zero = torch.zeros((), device=device, dtype=zero_dtype)
            return (
                zero,
                zero.detach(),
                zero.detach(),
                torch.zeros((), device=device, dtype=torch.long),
                zero.detach(),
            )

        # KL on common.
        if common_s.numel() > 0:
            student_common = student_chunks[:, :, common_s]
            teacher_common = teacher_chunks[:, :, common_t]
            if self.reverse_kl:
                kl_per_elem = torch.nn.functional.kl_div(
                    teacher_common, student_common, reduction="none", log_target=True
                )
            else:
                kl_per_elem = torch.nn.functional.kl_div(
                    student_common, teacher_common, reduction="none", log_target=True
                )
            kl_per_chunk = kl_per_elem.sum(dim=-1) * valid_chunk
            kl_common = kl_per_chunk.sum() / global_valid_chunks.to(
                kl_per_chunk.dtype
            ).clamp(min=1.0)
        else:
            kl_common = torch.zeros(
                (), device=device, dtype=zero_dtype, requires_grad=True
            )
            student_common = None
            teacher_common = None

        # L1 on uncommon.
        uncommon_topk = self.uncommon_topk
        if uncommon_s.numel() > 0 or uncommon_t.numel() > 0:
            student_unc = student_chunks[:, :, uncommon_s][valid_chunk]
            teacher_unc = teacher_chunks[:, :, uncommon_t][valid_chunk]
            n_valid = student_unc.shape[0]
            max_uncommon = min(
                student_unc.shape[-1], teacher_unc.shape[-1], uncommon_topk
            )
            if n_valid > 0 and max_uncommon > 0:
                student_unc_probs = student_unc.exp()
                teacher_unc_probs = teacher_unc.exp()
                if student_unc_probs.shape[-1] > max_uncommon:
                    student_sorted = torch.topk(
                        student_unc_probs, k=max_uncommon, dim=-1, largest=True
                    ).values
                else:
                    student_sorted = student_unc_probs.sort(
                        dim=-1, descending=True
                    ).values
                if teacher_unc_probs.shape[-1] > max_uncommon:
                    teacher_sorted = torch.topk(
                        teacher_unc_probs, k=max_uncommon, dim=-1, largest=True
                    ).values
                else:
                    teacher_sorted = teacher_unc_probs.sort(
                        dim=-1, descending=True
                    ).values
                min_len = min(student_sorted.shape[-1], teacher_sorted.shape[-1])
                student_sorted = student_sorted[:, :min_len]
                teacher_sorted = teacher_sorted[:, :min_len]
                l1_per_chunk = torch.nn.functional.l1_loss(
                    student_sorted, teacher_sorted, reduction="none"
                ).sum(dim=-1)
                l1_uncommon = l1_per_chunk.sum() / global_valid_chunks.to(
                    l1_per_chunk.dtype
                ).clamp(min=1.0)
            else:
                l1_uncommon = torch.zeros(
                    (), device=device, dtype=zero_dtype, requires_grad=True
                )
        else:
            l1_uncommon = torch.zeros(
                (), device=device, dtype=zero_dtype, requires_grad=True
            )

        # Top-1 accuracy on the common slice over valid chunks.
        with torch.no_grad():
            if student_common is not None and teacher_common is not None:
                s_common_valid = student_common[valid_chunk]
                t_common_valid = teacher_common[valid_chunk]
                matches = (
                    (s_common_valid.argmax(dim=-1) == t_common_valid.argmax(dim=-1))
                    .sum()
                    .float()
                )
                top1_acc = matches / valid_chunk.sum().float().clamp(min=1.0)
            else:
                top1_acc = torch.zeros((), device=device, dtype=zero_dtype)

        loss = (kl_common + l1_uncommon) * (T * T)
        return (
            loss,
            kl_common.detach(),
            l1_uncommon.detach(),
            valid_chunk.sum().detach(),
            top1_acc.detach(),
        )

    def _compute_ce(
        self,
        logits: torch.Tensor,
        data: BatchedDataDict[CrossTokenizerDistillationLossDataDict],
        global_valid_toks: torch.Tensor,
    ) -> torch.Tensor:
        """Next-token CE on the student side (TP/CP handled by the helpers)."""
        per_token_ce = student_next_token_ce(
            logits, input_ids=data["input_ids"], seq_index=data.get("seq_index")
        )
        label_mask = ce_label_mask(
            token_mask=data["token_mask"],
            sample_mask=data["sample_mask"],
            ce_seq_len=per_token_ce.shape[1],
            dtype=per_token_ce.dtype,
        )
        return masked_mean(
            per_token_ce, label_mask, global_normalization_factor=global_valid_toks
        )