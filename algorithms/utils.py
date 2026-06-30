"""Shared utility functions for RL algorithms in Project Dockyard.

Contains:
  set_seed                           — reproducible seeding
  get_tokenizer                      — HF tokenizer with chat-template config
  calculate_baseline_and_std_per_prompt — per-prompt GRPO baseline
  calculate_kl                       — re-exported from loss/utils
  get_gdpo_reward_component_keys     — detect named reward/<name> component keys
  mask_out_neg_inf_logprobs          — handle top-k/p sampling mask mismatch
  maybe_pad_last_batch               — pad validation batches for DP
  print_performance_metrics          — throughput / FLOP metrics to stdout
  log_generation_metrics_to_wandb    — vLLM timeline metrics to W&B
  surpress_user_warnings             — decorator
"""

import math
import random
import warnings
from functools import partial, wraps
from typing import Any, Optional
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from dockyard_rl.algorithms.loss.utils import calculate_kl  # re-export
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

# Re-exports
__all__ = [
    "set_seed",
    "get_tokenizer",
    "calculate_baseline_and_std_per_prompt",
    "calculate_kl",
    "get_gdpo_reward_component_keys",
    "mask_out_neg_inf_logprobs",
    "maybe_pad_last_batch",
    "print_performance_metrics",
    "log_generation_metrics_to_wandb",
    "surpress_user_warnings",
]

# Seeding
def set_seed(seed: int) -> None:
    """Set the seed for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# GDPO reward component detection and GRPO baseline calculation
def get_gdpo_reward_component_keys(batch) -> list[str]:
    """Return batch keys that are named reward components (e.g. reward/correctness) in sorted order."""
    return sorted(
        k for k in batch.keys() if isinstance(k, str) and k.startswith("reward/")
    )

# Per-prompt baseline and std calculation for GRPO-style advantage estimation
def calculate_baseline_and_std_per_prompt(
    prompts:                  torch.Tensor,
    rewards:                  torch.Tensor,
    valid_mask:               torch.Tensor,
    leave_one_out_baseline:   bool = True,
    std_rewards:              Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-(prompt, response) baselines and std for advantage estimation.

    Samples set to 0 in valid_mask are excluded from the baseline calculation.

    Args:
        prompts:                (batch, seq) or (batch,) — identifies each sample's prompt.
        rewards:                (batch,) — float-valued rewards.
        valid_mask:             (batch,) — 0/1 where 0 = ignore, 1 = keep.
        leave_one_out_baseline: If True, compute RLOO-style unbiased baseline.
        std_rewards:            (batch,) optional separate reward used only for the
                                std term; defaults to `rewards`. Lets DAPO compute
                                std on the raw task metric for dynamic-sampling
                                filtering while the baseline stays on the shaped reward.

    Returns:
        (baseline, std) — both (batch,) on the same device as rewards.
    """
    if std_rewards is None:
        std_rewards = rewards
    unique_prompts = torch.unique(prompts, dim=0)

    baseline    = torch.zeros_like(rewards)
    sq_baseline = torch.zeros_like(rewards)
    std         = torch.zeros_like(rewards)

    device_ordinal = rewards.get_device()
    reward_device  = (
        torch.device("cpu")
        if device_ordinal == -1
        else torch.device(f"cuda:{device_ordinal}")
    )

    for i in range(len(unique_prompts)):
        is_matching = (prompts == unique_prompts[i]).all(1)
        prompt_idx  = torch.arange(len(prompts), device=reward_device)[is_matching]

        if leave_one_out_baseline:
            baseline_mask_matrix = (
                1 - torch.eye(len(prompt_idx))
            ).to(reward_device)
        else:
            baseline_mask_matrix = torch.ones(
                (len(prompt_idx), len(prompt_idx))
            ).to(reward_device)

        if valid_mask[prompt_idx].sum() <= 1:
            # No valid responses; set baseline = reward to zero out gradient.
            baseline[prompt_idx] = rewards[prompt_idx]
        else:
            num_valid = (
                valid_mask[prompt_idx].float().sum()
                - int(leave_one_out_baseline)
            )
            prompt_baseline = (
                torch.matmul(
                    baseline_mask_matrix,
                    rewards[prompt_idx] * valid_mask[prompt_idx],
                )
                / num_valid
            )
            # std term uses std_rewards (== rewards unless a separate tensor was
            # passed for DAPO dynamic-sampling filtering).
            std_prompt_baseline = (
                prompt_baseline
                if std_rewards is rewards
                else torch.matmul(
                    baseline_mask_matrix,
                    std_rewards[prompt_idx] * valid_mask[prompt_idx],
                )
                / num_valid
            )
            std_prompt_baseline_sq = (
                torch.matmul(
                    baseline_mask_matrix,
                    torch.pow(std_rewards[prompt_idx], 2) * valid_mask[prompt_idx],
                )
                / num_valid
            )
            baseline[prompt_idx]    = prompt_baseline
            sq_baseline[prompt_idx] = std_prompt_baseline_sq
            std[prompt_idx]         = (
                (
                    (std_prompt_baseline_sq - std_prompt_baseline.square())
                    * (num_valid / (num_valid - 1))
                )
                .sqrt()
                .nan_to_num(0)
            )

    return baseline, std

# Logprob utilities
def mask_out_neg_inf_logprobs(
    logprobs:      torch.Tensor,
    mask:          torch.Tensor,
    logprobs_name: str,
) -> torch.Tensor:
    """Zero-out and mask positions with -inf log-probabilities.

    vLLM samples from a top-k/p filtered distribution; during training the
    policy may assign -inf to a token that was sampled if the distribution
    has shifted.  Masking these positions prevents NaN in the loss.
    """
    is_neginf   = torch.isinf(logprobs)
    neginf_count = (is_neginf & mask.bool()).sum().item()
    if neginf_count > 0:
        print(
            f"[WARNING]: {neginf_count}/{int(mask.sum().item())} valid tokens "
            f"have -inf in {logprobs_name} (policy top-k/top-p mismatch). "
            "Masking out these positions."
        )

    mask      = mask * (~is_neginf).float()
    logprobs  = torch.where(mask.bool(), logprobs, 0.0)
    return logprobs

# Batch padding
def maybe_pad_last_batch(batch: BatchedDataDict, dp_size: int, mbs: int) -> BatchedDataDict:
    """Pad the last validation batch so its size is divisible by (mbs × dp_size).

    Sample_mask is set to 0 for padded rows so they are excluded from loss.
    """
    min_padding = (
        math.ceil(batch.size / (mbs * dp_size)) * mbs * dp_size
    ) - batch.size
    if min_padding <= 0:
        return batch

    print(f"Padding last validation batch with {min_padding} padding samples")

    def _pad(tensor: torch.Tensor, dim1: bool = False) -> torch.Tensor:
        last = tensor[-1].unsqueeze(0)
        rep  = last.repeat(min_padding, 1) if dim1 else last.repeat(min_padding)
        return torch.cat([tensor, rep])

    batch["input_ids"]     = _pad(batch["input_ids"], dim1=True)
    batch["input_lengths"] = _pad(batch["input_lengths"])

    if "token_mask" in batch:
        batch["token_mask"] = _pad(batch["token_mask"], dim1=True)

    # Padding rows have sample_mask=0 so they don't contribute to the loss.
    batch["sample_mask"] = torch.cat([
        batch["sample_mask"],
        torch.zeros_like(batch["sample_mask"][-1]).unsqueeze(0).repeat(min_padding),
    ])

    if "reference_policy_logprobs" in batch:
        batch["reference_policy_logprobs"] = _pad(
            batch["reference_policy_logprobs"], dim1=True
        )

    return batch

# Tokenizer factory and performance metrics printing
def surpress_user_warnings(f):  # type: ignore[no-untyped-def]
    @wraps(f)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            output = f(*args, **kwargs)
        return output
    return wrapper

def get_tokenizer(
    tokenizer_config: dict,
    get_processor: bool = False,
) -> PreTrainedTokenizerBase:
    """Initialise a tokenizer (or processor) from a config dict.

    Args:
        tokenizer_config: Dict with keys:
            name (required):              HF model name or path.
            chat_template (optional):     None → passthrough; "default" → HF default;
                                          path ending .jinja → load from file;
                                          any other str → used as Jinja2 template.
            chat_template_kwargs (opt):   Dict of kwargs partially applied to
                                          tokenizer.apply_chat_template().
        get_processor:    If True, return an AutoProcessor instead.

    Returns:
        Configured tokenizer or processor instance.
    """
    # Deferred import to keep circular-import surface small.
    try:
        from dockyard_rl.data.chat_templates import COMMON_CHAT_TEMPLATES
    except ImportError:
        COMMON_CHAT_TEMPLATES = None  # type: ignore[assignment]

    processor = None
    if get_processor:
        processor  = AutoProcessor.from_pretrained(
            tokenizer_config["name"], trust_remote_code=True, use_fast=True
        )
        tokenizer  = processor.tokenizer
    else:
        tokenizer  = AutoTokenizer.from_pretrained(
            tokenizer_config["name"], trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "chat_template" in tokenizer_config:
        tpl = tokenizer_config["chat_template"]
        if tpl is None:
            print("Using passthrough chat template")
            if COMMON_CHAT_TEMPLATES is not None:
                tokenizer.chat_template = (
                    COMMON_CHAT_TEMPLATES.passthrough_prompt_response
                )
        elif str(tpl).lower() == "default":
            print("Using tokenizer's default chat template")
        elif str(tpl).endswith(".jinja"):
            print(f"Loading chat template from file: {tpl}")
            with open(tpl) as f:
                tokenizer.chat_template = f.read()
        else:
            print("Using custom chat template")
            tokenizer.chat_template = tpl
    else:
        print("No chat template provided, using tokenizer's default")

    if (
        "chat_template_kwargs" in tokenizer_config
        and tokenizer_config["chat_template_kwargs"] is not None
    ):
        assert isinstance(tokenizer_config["chat_template_kwargs"], dict), (
            "chat_template_kwargs should be a dictionary"
        )
        tokenizer.apply_chat_template = partial(
            tokenizer.apply_chat_template,
            **tokenizer_config["chat_template_kwargs"],
        )

    if processor is not None:
        # Forward special tokens so policy workers can use the processor.
        processor.pad_token     = tokenizer.pad_token
        processor.eos_token     = tokenizer.eos_token
        processor.bos_token     = tokenizer.bos_token
        processor.pad_token_id  = tokenizer.pad_token_id
        processor.eos_token_id  = tokenizer.eos_token_id
        processor.bos_token_id  = tokenizer.bos_token_id
        processor.name_or_path  = tokenizer.name_or_path
        if (
            not getattr(processor, "chat_template", None)
            and getattr(tokenizer, "chat_template", None)
        ):
            processor.chat_template = tokenizer.chat_template

    return tokenizer if processor is None else processor

# Performance metrics printing
def print_performance_metrics(
    train_results:   dict[str, float],
    metrics:         dict[str, Any],
    timing_metrics:  dict[str, float],
    master_config:   dict,
) -> dict[str, float]:
    """Print throughput and FLOP metrics for a GRPO training step.

    Returns a dict of scalar metrics suitable for W&B logging.
    """
    def visualize_per_worker_load(
        per_worker_token_counts: dict[int, int],
    ) -> float:
        counts_list   = [v for _, v in sorted(per_worker_token_counts.items())]
        max_count     = max(counts_list)
        load_ratios   = [v / max_count for v in counts_list]
        bar_length    = 20
        max_rows      = 1000
        print("  • Visualizing Token Imbalance per Generation Worker:")
        for i in range(min(len(counts_list), max_rows)):
            filled = int(load_ratios[i] * bar_length)
            empty  = bar_length - filled
            print(
                f"    - Generated Tokens from Worker {i:3.0f}:"
                f"{'█' * filled}{'░' * empty}"
                f" Count: {counts_list[i] / 1000:.1f}K"
            )
        est_idle = 1 - sum(load_ratios) / len(load_ratios)
        print(f"  • Average Token Imbalance: {100 * est_idle:.2f}%")
        return est_idle

    print("\n📋 Performance Metrics:")
    performance_metrics: dict[str, float] = {}

    if "per_worker_token_counts" in metrics:
        raw = metrics["per_worker_token_counts"]
        if isinstance(raw, list):
            merged: dict[int, int] = {}
            for tm in raw:
                for worker_idx, count in tm.items():
                    merged[worker_idx] = merged.get(worker_idx, 0) + count
            per_worker = merged
        elif isinstance(raw, dict):
            per_worker = raw
        else:
            per_worker = None

        if per_worker is not None:
            avg_imbalance = visualize_per_worker_load(per_worker)
            performance_metrics["average_token_imbalance"] = avg_imbalance

    if "mean_total_tokens_per_sample" in metrics:
        print(
            f"  • Mean Total Tokens per Sample: "
            f"{metrics['mean_total_tokens_per_sample']:.2f}"
        )

    policy_logprobs_time  = timing_metrics["policy_and_reference_logprobs"]
    policy_training_time  = timing_metrics["policy_training"]
    total_time            = timing_metrics["total_step_time"]
    refit_time = (
        timing_metrics.get("weight_sync")
        or timing_metrics.get("prepare_for_generation/total", 0.0)
    )

    if "generation" in timing_metrics:
        generation_time = timing_metrics["generation"]
    else:
        generation_time = (
            timing_metrics.get("exposed_generation", 0.0)
            + policy_logprobs_time
            + policy_training_time
        )

    num_nodes      = master_config.cluster["num_nodes"]
    gpus_per_node  = master_config.cluster["gpus_per_node"]
    total_num_gpus = num_nodes * gpus_per_node

    inference_nodes    = master_config.policy["generation"].get(
        "inference_fleet", {}
    ).get("num_nodes", 1)
    inference_gpus_per = master_config.policy["generation"].get(
        "inference_fleet", {}
    ).get("gpus_per_node", gpus_per_node)
    generation_num_gpus = inference_nodes * inference_gpus_per
    training_num_gpus   = total_num_gpus - generation_num_gpus

    grpo_config          = master_config.grpo
    num_samples_per_step = (
        grpo_config["num_prompts_per_step"]
        * grpo_config["num_generations_per_prompt"]
    )

    if (
        "async_grpo" in grpo_config
        and grpo_config["async_grpo"]["enabled"]
    ):
        exposed_gen_time = timing_metrics.get("exposed_generation", 0.0)
        idle_ratio = (
            0
            if exposed_gen_time > 0.1
            else exposed_gen_time / max(
                policy_training_time
                + policy_logprobs_time
                + exposed_gen_time
                + refit_time,
                1e-8,
            )
        )
        print(f"  • Training Worker Idle Time Ratio: {100 * idle_ratio:.2f}%")
        performance_metrics["training_worker_idle_time_ratio"] = idle_ratio

    total_num_tokens = metrics["total_num_tokens"]

    e2e_samples_per_sec_per_gpu = num_samples_per_step / total_time / total_num_gpus
    e2e_tokens_per_sec_per_gpu  = total_num_tokens / total_time / total_num_gpus
    policy_train_tps_per_gpu    = total_num_tokens / policy_training_time / training_num_gpus
    policy_logprobs_tps_per_gpu = total_num_tokens / policy_logprobs_time / training_num_gpus
    training_group_tps_per_gpu  = (
        total_num_tokens
        / (policy_training_time + policy_logprobs_time)
        / training_num_gpus
    )
    generation_tps_per_gpu      = total_num_tokens / generation_time / max(generation_num_gpus, 1)

    print("  • Throughputs (per GPU):")
    print(f"    - E2E (Samples/sec/gpu): {e2e_samples_per_sec_per_gpu:.2f}")
    print(f"    - E2E (Tokens/sec/gpu):  {e2e_tokens_per_sec_per_gpu:.2f}")
    print(f"    - Policy Training (Tokens/sec/gpu): {policy_train_tps_per_gpu:.2f}")
    print(f"    - Policy+Ref Logprobs (Tokens/sec/gpu): {policy_logprobs_tps_per_gpu:.2f}")
    print(f"    - Training Worker Group (Tokens/sec/gpu): {training_group_tps_per_gpu:.2f}")
    print(f"    - Generation Worker Group (Tokens/sec/gpu): {generation_tps_per_gpu:.2f}")

    if "total_flops" in train_results:
        total_tflops  = train_results["total_flops"] / policy_training_time / 1e12
        num_ranks     = train_results["num_ranks"]
        print(
            f"  • Training FLOPs: {total_tflops:.2f} TFLOPS "
            f"({total_tflops / num_ranks:.2f} TFLOPS per rank)",
            flush=True,
        )
        performance_metrics["train_flops_per_gpu"] = total_tflops / num_ranks
        if "theoretical_tflops" in train_results:
            theoretical = train_results["theoretical_tflops"]
            mfu = total_tflops / theoretical
            print(
                f"  • Training Model Floating Point Utilization: "
                f"{100 * mfu:.2f}%",
                flush=True,
            )
            performance_metrics["train_fp_utilization"] = mfu

    if "per_worker_token_counts" in metrics:
        del metrics["per_worker_token_counts"]

    performance_metrics.update({
        "samples_per_sec":                             e2e_samples_per_sec_per_gpu * total_num_gpus,
        "tokens_per_sec":                              e2e_tokens_per_sec_per_gpu * total_num_gpus,
        "samples_per_sec_per_gpu":                     e2e_samples_per_sec_per_gpu,
        "tokens_per_sec_per_gpu":                      e2e_tokens_per_sec_per_gpu,
        "policy_training_tokens_per_sec_per_gpu":      policy_train_tps_per_gpu,
        "policy_and_reference_logprobs_tokens_per_sec_per_gpu": policy_logprobs_tps_per_gpu,
        "training_worker_group_tokens_per_sec_per_gpu": training_group_tps_per_gpu,
        "generation_tokens_per_sec_per_gpu":           generation_tps_per_gpu,
        "training_worker_group_tokens_per_sec":        training_group_tps_per_gpu * training_num_gpus,
        "generation_tokens_per_sec":                   generation_tps_per_gpu * generation_num_gpus,
    })
    return performance_metrics

# W&B generation metrics logging
def log_generation_metrics_to_wandb(
    generation_logger_metrics: dict[str, dict[int, list[Any]]],
    step:               int,
    timeline_interval:  float,
    logger:             Any,
) -> None:
    """Log per-worker generation timeline metrics to W&B.

    Args:
        generation_logger_metrics: {metric_name: {dp_idx: [values]}}
        step:               Global training step.
        timeline_interval:  Seconds between timeline data points.
        logger:             dockyard_rl.utils.logger.Logger instance.
    """
    for metric_name in generation_logger_metrics:
        logger.log_plot_per_worker_timeline_metrics(
            generation_logger_metrics[metric_name],
            step=step,
            prefix="generation_metrics",
            name=metric_name,
            timeline_interval=timeline_interval,
        )