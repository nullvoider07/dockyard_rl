"""Pure-JAX GRPO/PPO loss and logprob, mirroring the torch reference.

Numerically mirrors ``algorithms/loss/utils.py`` (``masked_mean``,
``calculate_kl``) and the core of ``algorithms/loss/loss_functions.py``
``ClippedPGLossFn.__call__`` (verified against the torch source), so a single
shared fixture validates both. The scalar loss is differentiable under
``jax.value_and_grad``; the diagnostic metrics are returned as auxiliary
outputs (use ``has_aux=True``). The custom ``torch.autograd.Function`` backward
of the torch path is replaced by ``value_and_grad`` over the JAX params.

J3 scope = core GRPO path. These config branches raise ``NotImplementedError``
until J3b: ``sequence_level_importance_ratios``,
``truncated_importance_sampling_type in {"icepop","seq-mask-tis"}``,
``use_on_policy_kl_approximation``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp

Array = jax.Array


def masked_mean(
    tensor: Array,
    mask: Array,
    dim: Optional[int] = None,
    global_normalization_factor: Optional[Array] = None,
) -> Array:
    """JAX mirror of ``algorithms/loss/utils.py::masked_mean``."""
    masked = tensor * mask
    if dim is not None:
        numerator = jnp.sum(masked, axis=dim)
        denominator = jnp.maximum(jnp.sum(mask, axis=dim), 1)
        return numerator / denominator
    if global_normalization_factor is not None:
        return jnp.sum(masked) / jnp.maximum(global_normalization_factor, 1)
    return jnp.sum(masked) / jnp.maximum(jnp.sum(mask), 1)


def calculate_kl(
    logprobs: Array,
    logprobs_reference: Array,
    kl_type: str = "k3",
    input_clamp_value: Optional[float] = 20.0,
    output_clamp_value: Optional[float] = 10.0,
    importance_sampling_weights: Optional[Array] = None,
) -> Array:
    """JAX mirror of ``algorithms/loss/utils.py::calculate_kl`` (k1/k2/k3)."""
    log_ratio = logprobs - logprobs_reference
    if input_clamp_value is not None:
        log_ratio_clamped = jnp.clip(log_ratio, -input_clamp_value, input_clamp_value)
        if importance_sampling_weights is not None:
            importance_sampling_weights = jnp.where(
                log_ratio == log_ratio_clamped,
                importance_sampling_weights,
                jax.lax.stop_gradient(importance_sampling_weights),
            )
        log_ratio = log_ratio_clamped

    if kl_type == "k1":
        kl = log_ratio
    elif kl_type == "k2":
        kl = 0.5 * log_ratio**2
    elif kl_type == "k3":
        kl = jnp.exp(log_ratio) - log_ratio - 1.0
    else:
        raise ValueError(f"Unknown kl_type {kl_type!r}. Valid: 'k1','k2','k3'.")

    if importance_sampling_weights is not None:
        kl = importance_sampling_weights * kl

    if output_clamp_value is not None:
        kl = jnp.clip(kl, -output_clamp_value, output_clamp_value)
    return kl


def logprobs_from_logits(logits: Array, input_ids: Array) -> Array:
    """Token log-probabilities from logits (non-parallel branch).

    Mirrors the non-parallel path of
    ``distributed/model_utils.py::get_next_token_logprobs_from_logits``:
    drop the last position, ``log_softmax``, gather the next token. Returns
    shape ``[B, S-1]`` (no top-k/top-p filtering at training time).
    """
    logits = logits.astype(jnp.float32)
    logits_wo_last = logits[:, :-1]
    logprobs = jax.nn.log_softmax(logits_wo_last, axis=-1)
    next_tokens = input_ids[:, 1:]
    return jnp.take_along_axis(logprobs, next_tokens[..., None], axis=-1)[..., 0]


def _nan_to_num(x: Array) -> Array:
    # Mirrors torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) as used in the loss.
    return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _validate_cfg(cfg: Any) -> None:
    """Mirror the torch ClippedPGLossFn.__init__ validation asserts."""
    seq_level_is = _get(cfg, "sequence_level_importance_ratios", False)
    token_level = bool(_get(cfg, "token_level_loss", True))
    if seq_level_is and token_level:
        raise ValueError("sequence_level_importance_ratios is mutually exclusive with token_level_loss")

    tis_type = _get(cfg, "truncated_importance_sampling_type", None)
    if tis_type is not None:
        if not _get(cfg, "use_importance_sampling_correction", False):
            raise ValueError("truncated IS requires use_importance_sampling_correction=True")
        if tis_type not in ("tis", "icepop", "seq-mask-tis"):
            raise ValueError(f"Invalid truncated IS type: {tis_type!r}")
        ratio = _get(cfg, "truncated_importance_sampling_ratio", None)
        if ratio is None or not ratio > 0:
            raise ValueError("truncated_importance_sampling_ratio must be positive")
        ratio_min = _get(cfg, "truncated_importance_sampling_ratio_min", None)
        if ratio_min is not None and ratio_min > ratio:
            raise ValueError(
                "truncated_importance_sampling_ratio_min must be <= "
                "truncated_importance_sampling_ratio"
            )
        if tis_type in ("icepop", "seq-mask-tis") and _get(cfg, "truncated_importance_sampling_ratio_min", None) is None:
            raise ValueError("truncated_importance_sampling_ratio_min required for icepop / seq-mask-tis")
        if tis_type == "seq-mask-tis" and seq_level_is:
            raise ValueError("seq-mask-tis is incompatible with sequence_level_importance_ratios=True")

    if _get(cfg, "use_cispo", False):
        if _get(cfg, "disable_ppo_ratio", False):
            raise ValueError("use_cispo is incompatible with disable_ppo_ratio")
        if _get(cfg, "force_on_policy_ratio", False):
            raise ValueError("use_cispo is incompatible with force_on_policy_ratio")
        if seq_level_is:
            raise ValueError("use_cispo is incompatible with sequence_level_importance_ratios")
        if _get(cfg, "ratio_clip_c", None) is not None:
            raise ValueError("use_cispo is incompatible with dual clipping (ratio_clip_c)")
        if not token_level:
            raise ValueError("use_cispo requires token_level_loss=True (LossType.TOKEN_LEVEL)")


def clipped_pg_loss(
    curr_logprobs: Array,
    data: Mapping[str, Array],
    global_valid_seqs: Array,
    global_valid_toks: Array,
    cfg: Any,
) -> tuple[Array, dict[str, Array]]:
    """Core GRPO/PPO clipped policy-gradient loss in JAX.

    Args mirror ``ClippedPGLossFn.__call__``: ``curr_logprobs`` is the model's
    next-token logprobs ``[B, S-1]``; ``data`` holds the un-sliced columns
    (``token_mask``, ``sample_mask``, ``advantages``, ``prev_logprobs``,
    ``generation_logprobs``, optional ``reference_policy_logprobs``,
    ``curr_logprobs_unfiltered``), sliced ``[:, 1:]`` here exactly as torch does.
    ``cfg`` is duck-typed over the torch ``ClippedPGLossConfig`` fields.

    Returns ``(loss, metrics)``; only ``loss`` is differentiated.
    """
    _validate_cfg(cfg)

    ratio_clip_min = _get(cfg, "ratio_clip_min", 0.2)
    ratio_clip_max = _get(cfg, "ratio_clip_max", 0.2)
    ratio_clip_c = _get(cfg, "ratio_clip_c", None)
    kl_penalty = _get(cfg, "reference_policy_kl_penalty", 0.0)
    kl_type = _get(cfg, "reference_policy_kl_type", "k3")
    kl_in_clamp = _get(cfg, "kl_input_clamp_value", 20.0)
    kl_out_clamp = _get(cfg, "kl_output_clamp_value", 10.0)
    use_is = _get(cfg, "use_importance_sampling_correction", False)
    tis_type = _get(cfg, "truncated_importance_sampling_type", None)
    tis_ratio = _get(cfg, "truncated_importance_sampling_ratio", None)
    tis_ratio_min = _get(cfg, "truncated_importance_sampling_ratio_min", None)
    use_on_policy_kl = _get(cfg, "use_on_policy_kl_approximation", False)
    seq_level_is = _get(cfg, "sequence_level_importance_ratios", False)
    force_on_policy = _get(cfg, "force_on_policy_ratio", False)
    disable_ppo_ratio = _get(cfg, "disable_ppo_ratio", False)
    token_level = bool(_get(cfg, "token_level_loss", True))
    use_cispo = _get(cfg, "use_cispo", False)

    token_mask = data["token_mask"][:, 1:]
    sample_mask = data["sample_mask"]
    advantages = data["advantages"][:, 1:]
    generation_logprobs = data["generation_logprobs"][:, 1:]

    if force_on_policy:
        prev_logprobs = jax.lax.stop_gradient(curr_logprobs)
    else:
        prev_logprobs = data["prev_logprobs"][:, 1:]

    mask = token_mask * sample_mask[:, None]

    # --- Staleness / diagnostic metrics ---
    lp_error = jnp.abs(generation_logprobs - prev_logprobs)
    mult_prob_error = masked_mean(jnp.exp(lp_error * mask), mask, global_normalization_factor=global_valid_toks)
    gen_kl_error = masked_mean(
        calculate_kl(generation_logprobs, prev_logprobs, kl_type, None, None),
        mask, global_normalization_factor=global_valid_toks,
    )
    policy_kl_error = masked_mean(
        calculate_kl(prev_logprobs, generation_logprobs, kl_type, None, None),
        mask, global_normalization_factor=global_valid_toks,
    )
    log_mixture = jnp.log(0.5 * jnp.exp(prev_logprobs) + 0.5 * jnp.exp(generation_logprobs))
    kl_prev_to_mix = jnp.exp(prev_logprobs - log_mixture) - (prev_logprobs - log_mixture) - 1
    kl_gen_to_mix = jnp.exp(generation_logprobs - log_mixture) - (generation_logprobs - log_mixture) - 1
    js_divergence_error = masked_mean(
        0.5 * kl_prev_to_mix + 0.5 * kl_gen_to_mix, mask, global_normalization_factor=global_valid_toks
    )

    # --- KL regularization ---
    # KL samples come from the optimized policy, so the KL loss carries the
    # score-function gradient through the sampling probability (mirror of the
    # torch ClippedPGLossFn; https://arxiv.org/abs/2506.09477v1). The on-policy
    # ratio provides it; otherwise the straight-through exp(x - sg(x)) keeps the
    # gradient with forward value 1.
    if kl_penalty != 0:
        reference_policy_logprobs = data["reference_policy_logprobs"][:, 1:]
        curr_unfiltered = data.get("curr_logprobs_unfiltered", curr_logprobs)
        if use_on_policy_kl:
            kl_iw = jnp.exp(curr_unfiltered - generation_logprobs)
        else:
            kl_iw = jnp.exp(curr_unfiltered - jax.lax.stop_gradient(curr_unfiltered))
        kl_iw = _nan_to_num(kl_iw)
        kl_tok = kl_penalty * calculate_kl(
            curr_unfiltered, reference_policy_logprobs, kl_type, kl_in_clamp, kl_out_clamp,
            importance_sampling_weights=kl_iw,
        )
        if token_level:
            kl = masked_mean(kl_tok, mask, global_normalization_factor=global_valid_toks)
        else:
            kl = masked_mean(
                masked_mean(kl_tok, token_mask, dim=-1), sample_mask,
                global_normalization_factor=global_valid_seqs,
            )
    else:
        kl = jnp.asarray(0.0, dtype=curr_logprobs.dtype)

    # --- Probability ratio ---
    if force_on_policy:
        log_ratios = curr_logprobs - jax.lax.stop_gradient(curr_logprobs)
        ratios = jnp.exp(log_ratios)
        ratios_clamped = ratios
    elif not disable_ppo_ratio:
        log_ratios = curr_logprobs - prev_logprobs
        if seq_level_is:
            seq_log_ratio_mean = masked_mean(log_ratios, token_mask, dim=-1)[:, None]
            seq_ratio = jnp.exp(seq_log_ratio_mean)
            ratios = jnp.broadcast_to(seq_ratio, advantages.shape)
        else:
            ratios = jnp.exp(log_ratios)
        ratios_clamped = jnp.clip(ratios, 1.0 - ratio_clip_min, 1.0 + ratio_clip_max)
    else:
        ratios = curr_logprobs
        ratios_clamped = curr_logprobs

    if use_cispo:
        # CISPO (arXiv:2506.13585): stop-gradient clipped IS weight on a
        # REINFORCE-style surrogate; gradient flows only through curr_logprobs.
        clip_loss = -advantages * jax.lax.stop_gradient(ratios_clamped) * curr_logprobs
    else:
        loss1 = -advantages * ratios
        loss2 = -advantages * ratios_clamped
        clip_loss = jnp.maximum(loss1, loss2)

    if ratio_clip_c is not None:
        if not ratio_clip_c > 1:
            raise ValueError(f"ratio_clip_c must exceed 1, got {ratio_clip_c}")
        loss3 = -advantages * ratio_clip_c
        clip_loss = jnp.where(advantages < 0, jnp.minimum(clip_loss, loss3), clip_loss)

    # --- Off-policy IS correction ---
    if seq_level_is:
        seq_lp_diff = jnp.sum((prev_logprobs - generation_logprobs) * mask, axis=-1)
        actor_iw = _nan_to_num(jax.lax.stop_gradient(jnp.exp(seq_lp_diff)))[:, None]
    else:
        actor_iw = _nan_to_num(jnp.exp(prev_logprobs - generation_logprobs))

    is_oob_ratio = jnp.asarray(0.0, dtype=curr_logprobs.dtype)
    if tis_ratio is not None:
        if tis_type == "tis":
            tis_min = 0.0 if tis_ratio_min is None else tis_ratio_min
            token_in_bounds = (
                (actor_iw <= tis_ratio) & (actor_iw >= tis_min)
            ).astype(curr_logprobs.dtype)
            is_oob_ratio = 1.0 - masked_mean(token_in_bounds, mask, global_normalization_factor=global_valid_toks)
            actor_iw = jnp.clip(actor_iw, tis_min, tis_ratio)
        elif tis_type == "icepop":
            token_kept = (actor_iw >= tis_ratio_min) & (actor_iw <= tis_ratio)
            is_oob_ratio = 1.0 - masked_mean(
                token_kept.astype(curr_logprobs.dtype), mask, global_normalization_factor=global_valid_toks
            )
            actor_iw = jnp.where(token_kept, actor_iw, jnp.zeros_like(actor_iw))
        elif tis_type == "seq-mask-tis":
            log_is_ratio = _nan_to_num(prev_logprobs - generation_logprobs)
            seq_log_is_ratio_mean = masked_mean(log_is_ratio, token_mask, dim=-1)
            seq_geomean_is = jax.lax.stop_gradient(jnp.exp(seq_log_is_ratio_mean))
            seq_kept = (seq_geomean_is >= tis_ratio_min) & (seq_geomean_is <= tis_ratio)
            seq_kept_f = seq_kept.astype(curr_logprobs.dtype)
            is_oob_ratio = 1.0 - masked_mean(seq_kept_f, sample_mask, global_normalization_factor=global_valid_seqs)
            actor_iw = actor_iw * seq_kept_f[:, None]

    importance_weights = actor_iw if use_is else jnp.ones_like(prev_logprobs)

    if token_level:
        actor_loss = masked_mean(
            importance_weights * clip_loss, mask, global_normalization_factor=global_valid_toks
        )
    else:
        actor_loss = masked_mean(
            masked_mean(importance_weights * clip_loss, token_mask, dim=-1), sample_mask,
            global_normalization_factor=global_valid_seqs,
        )

    if seq_level_is:
        sample_importance_ratio = masked_mean(
            actor_iw[:, 0], sample_mask, global_normalization_factor=global_valid_seqs
        )
    else:
        sample_importance_ratio = masked_mean(actor_iw, mask, global_normalization_factor=global_valid_toks)
    seq_entropy_approx = -masked_mean(
        jnp.exp(curr_logprobs - generation_logprobs) * curr_logprobs,
        mask, global_normalization_factor=global_valid_toks,
    )

    loss = actor_loss + kl

    probs_ratio = masked_mean(jax.lax.stop_gradient(ratios), mask, global_normalization_factor=global_valid_toks)
    probs_ratio_clamped = masked_mean(
        jax.lax.stop_gradient(ratios_clamped), mask, global_normalization_factor=global_valid_toks
    )
    metrics: dict[str, Array] = {
        "loss": jax.lax.stop_gradient(loss),
        "probs_ratio": probs_ratio,
        "probs_ratio_clamped": probs_ratio_clamped,
        "kl_penalty": (kl / kl_penalty) if kl_penalty else jnp.asarray(0.0),
        "token_mult_prob_error": mult_prob_error,
        "gen_kl_error": gen_kl_error,
        "policy_kl_error": policy_kl_error,
        "js_divergence_error": js_divergence_error,
        "sampling_importance_ratio": sample_importance_ratio,
        "num_valid_samples": jnp.sum(sample_mask),
        "approx_entropy": seq_entropy_approx,
        "is_oob_ratio": is_oob_ratio,
    }
    return loss, metrics
