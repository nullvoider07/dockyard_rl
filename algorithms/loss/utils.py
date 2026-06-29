"""Loss utility functions for Project Dockyard.

Provides masked_mean and calculate_kl, used by every loss function and by the
advantage estimator, plus prepare_loss_input — the per-input-type dispatcher the
DTensor policy worker uses to turn raw student logits into the kwargs each loss
function consumes.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING
import torch

from dockyard_rl.algorithms.loss.interfaces import LossInputType

if TYPE_CHECKING:
    from dockyard_rl.algorithms.loss.interfaces import LossFunction
    from dockyard_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
    from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

def masked_mean(
    tensor:                     torch.Tensor,
    mask:                       torch.Tensor,
    dim:                        Optional[int] = None,
    global_normalization_factor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the mean of tensor over masked positions.

    Args:
        tensor: Values to average.
        mask:   Binary mask (1 = valid, 0 = pad).  Same shape as tensor
                or broadcastable.
        dim:    If provided, reduce along this dimension only and return a
                tensor.  If None, reduce to a scalar.
        global_normalization_factor:
                When provided, divides by this value instead of the local
                mask sum.  Used for globally-normalised losses across
                microbatches (pass the total valid token / sequence count
                across all DP ranks).

    Returns:
        Scalar or reduced tensor.
    """
    masked = tensor * mask
    if dim is not None:
        numerator   = masked.sum(dim=dim)
        denominator = mask.sum(dim=dim).clamp(min=1)
        return numerator / denominator

    if global_normalization_factor is not None:
        return masked.sum() / global_normalization_factor.clamp(min=1)

    return masked.sum() / mask.sum().clamp(min=1)

def calculate_kl(
    logprobs:           torch.Tensor,
    logprobs_reference: torch.Tensor,
    kl_type:            str = "k3",
    input_clamp_value:  Optional[float] = 20.0,
    output_clamp_value: Optional[float] = 10.0,
) -> torch.Tensor:
    """Compute a per-token KL divergence approximation.

    Supports three estimators from http://joschu.net/blog/kl-appox.html:

    k1  — log_ratio (first-order Taylor approximation)
    k2  — 0.5 * log_ratio²  (symmetric, second-order)
    k3  — exp(log_ratio) - log_ratio - 1  (Schulman approximation, always ≥ 0)

    Args:
        logprobs:           Log-probabilities from the current policy.
        logprobs_reference: Log-probabilities from the reference policy.
        kl_type:            One of "k1", "k2", "k3".
        input_clamp_value:  Clamp |log_ratio| before exponentiation to prevent
                            numerical overflow.  None to disable.
        output_clamp_value: Clamp the output KL values.  None to disable.

    Returns:
        Per-token KL tensor, same shape as logprobs.
    """
    log_ratio = logprobs - logprobs_reference

    if input_clamp_value is not None:
        log_ratio = log_ratio.clamp(-input_clamp_value, input_clamp_value)

    if kl_type == "k1":
        kl = log_ratio
    elif kl_type == "k2":
        kl = 0.5 * log_ratio ** 2
    elif kl_type == "k3":
        kl = torch.exp(log_ratio) - log_ratio - 1.0
    else:
        raise ValueError(
            f"Unknown kl_type {kl_type!r}. Valid options: 'k1', 'k2', 'k3'."
        )

    if output_clamp_value is not None:
        kl = kl.clamp(-output_clamp_value, output_clamp_value)

    return kl


def prepare_loss_input(
    logits:                 torch.Tensor,
    data:                   "BatchedDataDict[Any]",
    loss_fn:                "LossFunction",
    vocab_parallel_rank:    Optional[int] = None,
    vocab_parallel_group:   Optional[torch.distributed.ProcessGroup] = None,
    context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    sampling_params:        Optional["TrainingSamplingParams"] = None,
) -> tuple[dict[str, Any], "BatchedDataDict[Any]"]:
    """Turn raw student logits into the per-input-type kwargs a loss fn consumes.

    Dispatches on ``loss_fn.input_type``. The heavy logprob / distillation
    helpers and the cross-tokenizer keystone are imported inside their branches:
    ``mask_out_neg_inf_logprobs`` lives in ``dockyard_rl.algorithms.utils``,
    which imports this module (``calculate_kl``), so a module-level import would
    be circular; and the cross-tokenizer teacher-logit rebuild is M3-deferred.

    Args:
        logits: Student logits ``[B, T, V]`` from the worker forward (or, on the
            linear-CE-fusion path, precomputed next-token logprobs ``[B, T-1]``).
        data: Microbatch data. Mutated in place when the LOGPROB branch writes
            ``curr_logprobs_unfiltered`` for the reference-policy KL penalty.
        loss_fn: The loss function; only ``input_type`` (and a few optional
            attributes read via ``hasattr``) are consulted here.
        vocab_parallel_rank / vocab_parallel_group / context_parallel_group:
            Parallelism groups forwarded to the logprob / distillation helpers.
        sampling_params: Only used by the LOGPROB branch (top-k/top-p filtering,
            currently ClippedPGLossFn).

    Returns:
        ``(loss_input, data)`` — the kwargs dict for ``loss_fn`` and the
        (possibly updated) data dict.

    Raises:
        NotImplementedError: ``DRAFT`` input prep is not ported (Megatron-coupled
            upstream; the draft path is outside the cross-tokenizer scope).
        ValueError: unknown ``input_type``.
    """
    if loss_fn.input_type == LossInputType.LOGIT:
        loss_input: dict[str, Any] = {"logits": logits}

    elif loss_fn.input_type == LossInputType.LOGPROB:
        from dockyard_rl.distributed.model_utils import (
            get_next_token_logprobs_from_logits,
        )
        from dockyard_rl.algorithms.logits_sampling_utils import (
            need_top_k_or_top_p_filtering,
        )
        from dockyard_rl.algorithms.utils import mask_out_neg_inf_logprobs

        # Linear CE fusion returns precomputed next-token logprobs (2D tensor);
        # the standard path computes them from 3D logits.
        if hasattr(loss_fn, "use_linear_ce_fusion") and loss_fn.use_linear_ce_fusion:
            logprobs = logits.to(torch.float32)
            logprobs = logprobs[:, : data["input_ids"].shape[1] - 1]
        else:
            logprobs = get_next_token_logprobs_from_logits(
                input_ids=data["input_ids"],
                next_token_logits=logits,
                seq_index=data.get("seq_index", None),
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
                sampling_params=sampling_params,
            )

        # top-k/top-p filtering for logprobs (currently ClippedPGLossFn only).
        if need_top_k_or_top_p_filtering(sampling_params):
            mask = data["token_mask"] * data["sample_mask"].unsqueeze(-1)
            logprobs = mask_out_neg_inf_logprobs(logprobs, mask[:, 1:], "curr_logprobs")
            # Unfiltered logprobs for the reference-policy KL penalty.
            if (
                hasattr(loss_fn, "reference_policy_kl_penalty")
                and loss_fn.reference_policy_kl_penalty != 0
            ):
                data["curr_logprobs_unfiltered"] = get_next_token_logprobs_from_logits(
                    input_ids=data["input_ids"],
                    next_token_logits=logits,
                    seq_index=data.get("seq_index", None),
                    vocab_parallel_rank=vocab_parallel_rank,
                    vocab_parallel_group=vocab_parallel_group,
                    context_parallel_group=context_parallel_group,
                    sampling_params=None,  # no filtering
                )

        loss_input = {"next_token_logprobs": logprobs}

    elif loss_fn.input_type == LossInputType.DISTILLATION:
        from dockyard_rl.distributed.model_utils import (
            get_distillation_topk_logprobs_from_logits,
        )

        calculate_entropy = loss_fn.zero_outside_topk and loss_fn.kl_type != "forward"
        student_topk_logprobs, teacher_topk_logprobs, H_all = (
            get_distillation_topk_logprobs_from_logits(
                student_logits=logits,
                teacher_topk_logits=data["teacher_topk_logits"],
                teacher_topk_indices=data["teacher_topk_indices"],
                zero_outside_topk=loss_fn.zero_outside_topk,
                calculate_entropy=calculate_entropy,
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
            )
        )
        loss_input = {
            "student_topk_logprobs": student_topk_logprobs,
            "teacher_topk_logprobs": teacher_topk_logprobs,
            "H_all": H_all,
        }

    elif loss_fn.input_type == LossInputType.DISTILLATION_CROSS_TOKENIZER:
        # Rebuild full-vocab teacher logits from the per-rank CUDA IPC handles
        # and do the shared CP-resolution both loss paths need; the loss fn does
        # the projection / chunk-average / KL reductions. The TP/CP groups are
        # derived from the student logits' device mesh. The keystone
        # prepare_xtoken_cross_tokenizer_loss_input lands with the M3 transport
        # layer; imported lazily so this dispatcher stays import-clean until then.
        from dockyard_rl.algorithms.x_token.loss_utils import (
            prepare_xtoken_cross_tokenizer_loss_input,
        )

        teacher_full_logits, student_logits_contig, align, tp_group, cp_group = (
            prepare_xtoken_cross_tokenizer_loss_input(
                logits,
                data,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
            )
        )
        loss_input = {
            "logits": logits,
            "teacher_full_logits": teacher_full_logits,
            "student_logits_contig": student_logits_contig,
            "align": align,
            "tp_group": tp_group,
            "cp_group": cp_group,
        }

    elif loss_fn.input_type == LossInputType.DRAFT:
        raise NotImplementedError(
            "DRAFT loss-input preparation is not ported: the upstream DRAFT "
            "branch is Megatron-coupled (roll_tensor / "
            "gather_from_tensor_model_parallel_region) and the speculative-decode "
            "draft path is outside the cross-tokenizer scope. Build a "
            "DTensor-native DRAFT branch when that path is taken up."
        )

    else:
        raise ValueError(f"Unknown loss function input type: {loss_fn.input_type}")

    return loss_input, data