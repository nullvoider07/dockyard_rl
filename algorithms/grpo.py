"""GRPO training algorithm for Project Dockyard.

Entry points:
  setup()            -- initialise all components.
  grpo_train()       -- synchronous GRPO training loop.
  async_grpo_train() -- asynchronous GRPO with replay buffer.
  validate()         -- validation pass.
"""

import gc
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NotRequired, Optional, TypedDict, cast, TYPE_CHECKING
import numpy as np
import ray
import torch
from pydantic import BaseModel
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dockyard_rl.algorithms.advantage_estimator import (
    GDPOAdvantageEstimator,
    GRPOAdvantageEstimator,
    ReinforcePlusPlusAdvantageEstimator,
)
from dockyard_rl.algorithms.async_utils import AsyncTrajectoryCollector, ReplayBuffer
from dockyard_rl.algorithms.loss import (
    ClippedPGLossConfig,
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from dockyard_rl.algorithms.loss.interfaces import LossFunction
from dockyard_rl.algorithms.reward_functions import (
    RewardShapingConfig,
    apply_invalid_action_penalty,
    apply_message_span_advantage_penalties,
    apply_reward_shaping,
)
from dockyard_rl.rewards.invalid_action import InvalidActionPenaltyConfig
from dockyard_rl.tool_protocol.protocol import (
    StructuredToolUseConfig,
    resolve_structured_tool_use,
)
from dockyard_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    get_gdpo_reward_component_keys,
    log_generation_metrics_to_wandb,
    print_performance_metrics,
    set_seed,
)
from dockyard_rl.data.collate_fn import rl_collate_fn
from dockyard_rl.data.dataloader import MultipleDataloaderWrapper
from dockyard_rl.data.datasets.utils import extract_necessary_env_names
from dockyard_rl.data_plane.interfaces import DataPlaneConfig
from dockyard_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from dockyard_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from dockyard_rl.models.generation.interfaces import GenerationInterface
from dockyard_rl.models.generation.sglang import SGLangConfig, SGLangGeneration
from dockyard_rl.models.generation.vllm import VllmConfig, VllmGeneration
from dockyard_rl.models.policy.lm_policy import Policy
from dockyard_rl.utils.checkpoint import CheckpointManager, CheckpointingConfig
from dockyard_rl.utils.logger import Logger, LoggerConfig, print_message_log_samples
from dockyard_rl.utils.memory_tracker import MemoryTracker
from dockyard_rl.utils.nsys import maybe_gpu_profile_step
from dockyard_rl.utils.timer import TimeoutChecker, Timer
from dockyard_rl.weight_sync import WeightSynchronizer, create_weight_synchronizer

# ============================================================
# Configuration
# ============================================================

class RewardScalingConfig(TypedDict):
    """Configure linear reward scaling with clamping.

    When `enabled` is True, each reward is clamped to [source_min, source_max]
    and linearly mapped to [target_min, target_max].
    """
    enabled:    bool
    source_min: NotRequired[float]
    source_max: NotRequired[float]
    target_min: NotRequired[float]
    target_max: NotRequired[float]

class AsyncGRPOConfig(TypedDict):
    enabled:                                 bool
    max_trajectory_age_steps:                int
    in_flight_weight_updates:                NotRequired[bool]
    recompute_kv_cache_after_weight_updates: NotRequired[bool]

class AdvEstimatorConfig(TypedDict):
    """Configuration for advantage estimator (grpo, gdpo, or reinforce_plus_plus)."""
    name:                       str
    normalize_rewards:          NotRequired[bool]
    use_leave_one_out_baseline: NotRequired[bool]
    minus_baseline:             NotRequired[bool]

class GRPOConfig(TypedDict):
    num_prompts_per_step:                       int
    num_generations_per_prompt:                 int
    max_num_epochs:                             int
    max_num_steps:                              int
    max_rollout_turns:                          int
    normalize_rewards:                          bool
    use_leave_one_out_baseline:                 bool
    val_period:                                 int
    val_batch_size:                             int | None
    val_at_start:                               bool
    val_at_end:                                 bool
    max_val_samples:                            int | None
    skip_reference_policy_logprobs_calculation: NotRequired[bool]
    seed:                                       int
    async_grpo:                                 NotRequired[AsyncGRPOConfig]
    overlong_filtering:                         NotRequired[bool]
    use_dynamic_sampling:                       bool
    dynamic_sampling_max_gen_batches:           NotRequired[int]
    batch_multiplier:                           NotRequired[float]
    reward_shaping:                             RewardShapingConfig
    reward_scaling:                             RewardScalingConfig
    # Native invalid-action / malformed-thinking reward penalty (#2656).
    invalid_action_penalty:                     NotRequired[InvalidActionPenaltyConfig | None]
    # Structured (Hermes) tool-use protocol. Disabled (default) is a strict
    # no-op: the fenced-text rollout path runs byte-identically.
    structured_tool_use:                        NotRequired[StructuredToolUseConfig | None]
    calculate_advantages_on_gpu:                NotRequired[bool]
    seq_logprob_error_threshold:                float | None
    adv_estimator:                              NotRequired[AdvEstimatorConfig]
    # When set, advantages are clamped to [advantage_clip_low, advantage_clip_high]
    # after normalization.
    advantage_clip_low:                         NotRequired[float | None]
    advantage_clip_high:                        NotRequired[float | None]

class GRPOSaveState(TypedDict):
    consumed_samples:   int
    current_step:       int
    current_epoch:      int
    total_steps:        int
    total_valid_tokens: int
    val_reward:         NotRequired[float]

def _default_grpo_save_state() -> GRPOSaveState:
    return {
        "consumed_samples":   0,
        "current_step":       0,
        "current_epoch":      0,
        "total_steps":        0,
        "total_valid_tokens": 0,
        "val_reward":         -99999999.0,
    }

class MasterConfig(BaseModel, extra="allow"):
    policy:        dict[str, Any]
    loss_fn:       ClippedPGLossConfig
    env:           dict[str, Any]
    data:          dict[str, Any]
    grpo:          GRPOConfig
    logger:        dict[str, Any]
    cluster:       ClusterConfig
    checkpointing: dict[str, Any]
    # Experience-data transfer plane for non-colocated trainer/generation
    # fleets. None (default) uses the colocated in-memory path; the
    # presharded *_from_meta driver dispatch that consumes it is the
    # sync-GRPO path (not built in this async system).
    data_plane:    Optional[DataPlaneConfig] = None

# ============================================================
# Setup & Initialisation
# ============================================================

def setup(
    master_config: MasterConfig,
    tokenizer:     PreTrainedTokenizerBase,
    dataset,
    val_dataset:   Optional[Any],
    processor:     Optional[AutoProcessor] = None,
) -> tuple:
    """Main entry point for setting up GRPO training.

    Returns:
        (policy, policy_generation, (train_cluster, inference_cluster),
         dataloader, val_dataloader, loss_fn, logger, checkpointer,
         grpo_save_state, master_config, weight_synchronizer)
    """
    setup_start_time = time.perf_counter()

    policy_config        = master_config.policy
    generation_config    = policy_config["generation"]
    loss_config: ClippedPGLossConfig = master_config.loss_fn
    env_configs          = master_config.env
    data_config          = master_config.data
    grpo_config          = master_config.grpo
    logger_config        = master_config.logger
    cluster_config       = master_config.cluster
    checkpointing_config = master_config.checkpointing


    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for GRPO"
    )

    set_seed(grpo_config["seed"])

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(cast(LoggerConfig, logger_config))
    logger.log_hyperparams(master_config.model_dump())

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(cast(CheckpointingConfig, checkpointing_config))
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    grpo_save_state: Optional[GRPOSaveState] = cast(
        Optional[GRPOSaveState],
        checkpointer.load_training_info(last_checkpoint_path),
    )
    if grpo_save_state is None:
        grpo_save_state = _default_grpo_save_state()

    # ==========================
    #           Data
    # ==========================
    num_prompts_per_step = grpo_config["num_prompts_per_step"]
    if data_config["use_multiple_dataloader"]:
        dataloader_batch_size = data_config["num_prompts_per_dataloader"]
    else:
        dataloader_batch_size = num_prompts_per_step

    batch_multiplier = grpo_config.get("batch_multiplier", 1)
    if grpo_config["use_dynamic_sampling"]:
        num_prompts_per_step  = int(num_prompts_per_step  * batch_multiplier)
        dataloader_batch_size = int(dataloader_batch_size * batch_multiplier)
    else:
        assert batch_multiplier == 1, (
            "batch_multiplier > 1 can only be used if use_dynamic_sampling=True"
        )

    if data_config["use_multiple_dataloader"]:
        assert num_prompts_per_step % dataloader_batch_size == 0, (
            "Expected num_prompts_per_step to be a multiple of "
            "num_prompts_per_dataloader, but got "
            f"{num_prompts_per_step} and {dataloader_batch_size}."
        )

    def init_train_dataloader(ds, suffix: str = ""):
        dl = StatefulDataLoader(
            ds,
            batch_size  = dataloader_batch_size,
            shuffle     = data_config["shuffle"],
            collate_fn  = rl_collate_fn,
            drop_last   = True,
            num_workers = data_config["num_workers"],
        )
        if last_checkpoint_path is not None:
            state = torch.load(
                os.path.join(last_checkpoint_path, f"train_dataloader{suffix}.pt")
            )
            dl.load_state_dict(state)
        return dl

    if data_config["use_multiple_dataloader"]:
        dataloaders = {}
        for task_name, task_dataset in dataset.items():
            dataloaders[task_name] = init_train_dataloader(task_dataset, f"_{task_name}")
            print(
                f"  ✓ Training dataloader {task_name} loaded with "
                f"{len(task_dataset)} samples",
                flush=True,
            )
        train_sample_count = sum(len(dl) for dl in dataloaders.values())
        dataloader = MultipleDataloaderWrapper(
            expected_num_prompts = num_prompts_per_step,
            data_config          = data_config,
            dataloaders          = dataloaders,
        )
    else:
        dataloader         = init_train_dataloader(dataset)
        train_sample_count = len(dataloader)
        print(
            f"  ✓ Training dataloader loaded with {train_sample_count} samples",
            flush=True,
        )

    val_dataloader: Optional[StatefulDataLoader] = None
    if (
        grpo_config["val_period"] > 0
        or grpo_config["val_at_start"]
        or grpo_config["val_at_end"]
    ):
        assert val_dataset is not None, (
            "Validation dataset is required if validation is enabled"
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size  = grpo_config["val_batch_size"],
            shuffle     = False,
            collate_fn  = rl_collate_fn,
            num_workers = data_config["num_workers"],
        )
        print(
            f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples",
            flush=True,
        )

    # ==========================
    #        Loss Function
    # ==========================
    loss_fn = ClippedPGLossFn(loss_config)

    if loss_config.force_on_policy_ratio:
        assert (
            grpo_config["num_prompts_per_step"]
            * grpo_config["num_generations_per_prompt"]
            == policy_config["train_global_batch_size"]
        ), (
            "force_on_policy_ratio requires "
            "train_global_batch_size == num_prompts_per_step * num_generations_per_prompt"
        )
        os.environ["DOCKYARD_IGNORE_TP_ACCURACY_CHECK"] = "1"
        print("  ✓ force_on_policy_ratio enabled")

    if grpo_config.get("skip_reference_policy_logprobs_calculation"):
        assert loss_config.reference_policy_kl_penalty == 0, (
            "grpo.skip_reference_policy_logprobs_calculation=True requires "
            "loss_fn.reference_policy_kl_penalty == 0"
        )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...", flush=True)
    colocated_inference = generation_config["colocated"]["enabled"]

    env_name_list = extract_necessary_env_names(data_config)
    rm_env_enabled = "reward_model" in env_name_list

    total_nodes = cluster_config["num_nodes"]
    if rm_env_enabled:
        rm_resource      = env_configs["reward_model"]["resources"]
        rm_nodes         = rm_resource["num_nodes"]
        rm_gpus_per_node = rm_resource["gpus_per_node"]
    else:
        rm_nodes         = 0
        rm_gpus_per_node = 0

    if total_nodes == 1:
        policy_nodes = total_nodes
    else:
        policy_nodes = total_nodes - rm_nodes
        assert policy_nodes > 0, (
            f"policy_nodes must be > 0, but got "
            f"policy_nodes:{policy_nodes} + rm_nodes:{rm_nodes} = total_nodes:{total_nodes}"
        )

    inference_nodes         = 0
    inference_gpus_per_node = 0

    if colocated_inference:
        if total_nodes == 1:
            policy_gpus_per_node = cluster_config["gpus_per_node"] - rm_gpus_per_node
            assert policy_gpus_per_node > 0, (
                "policy_gpus_per_node must be > 0 when cluster.num_nodes=1"
            )
        else:
            policy_gpus_per_node = cluster_config["gpus_per_node"]

        cluster = RayVirtualCluster(
            name                        = "grpo_policy_cluster",
            bundle_ct_per_node_list     = [policy_gpus_per_node] * policy_nodes,
            use_gpus                    = True,
            num_gpus_per_node           = policy_gpus_per_node,
            max_colocated_worker_groups = 2,
        )
        train_cluster     = cluster
        inference_cluster = cluster
        print(
            f"  ✓ Ray cluster for policy initialized with {policy_nodes} nodes",
            flush=True,
        )
    else:
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes         = policy_nodes

        inference_resources      = generation_config["colocated"]["resources"]
        inference_gpus_per_node  = inference_resources["gpus_per_node"]
        inference_nodes          = inference_resources["num_nodes"]

        if policy_nodes == 1:
            assert (
                inference_gpus_per_node is not None and inference_gpus_per_node > 0
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be > 0 "
                "when policy_nodes=1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            assert inference_nodes is None or inference_nodes == 1, (
                "policy.generation.colocated.resources.num_nodes must be 1 or null "
                f"when policy_nodes=1, but got {inference_nodes}."
            )
            inference_nodes = 1
            reward_gpus_to_subtract = (
                rm_gpus_per_node if total_nodes == 1 and rm_env_enabled else 0
            )
            train_gpus_per_node -= inference_gpus_per_node + reward_gpus_to_subtract
            assert train_gpus_per_node > 0, (
                "Not enough GPUs for training: "
                f"train_gpus_per_node={train_gpus_per_node} = "
                f"cluster_config['gpus_per_node']:{cluster_config['gpus_per_node']} "
                f"- inference_gpus_per_node:{inference_gpus_per_node}"
            )
        else:
            assert inference_nodes > 0, (
                "policy.generation.colocated.resources.num_nodes must be > 0 "
                f"when cluster.num_nodes > 1, but got {inference_nodes}."
            )
            assert (
                inference_gpus_per_node is not None
                and inference_gpus_per_node == cluster_config["gpus_per_node"]
            ), (
                "policy.generation.colocated.resources.gpus_per_node must equal "
                f"cluster.gpus_per_node when cluster.num_nodes > 1, but got "
                f"inference_gpus_per_node={inference_gpus_per_node}, "
                f"cluster.gpus_per_node={cluster_config['gpus_per_node']}."
            )
            train_nodes -= inference_nodes

        train_cluster = RayVirtualCluster(
            name                        = "grpo_train_cluster",
            bundle_ct_per_node_list     = [train_gpus_per_node] * train_nodes,
            use_gpus                    = True,
            num_gpus_per_node           = train_gpus_per_node,
            max_colocated_worker_groups = 1,
        )
        print(
            f"  ✓ Ray train cluster initialized with {train_nodes} nodes "
            f"with {train_gpus_per_node} GPUs per node",
            flush=True,
        )

        inference_cluster = RayVirtualCluster(
            name                        = "grpo_inference_cluster",
            bundle_ct_per_node_list     = [inference_gpus_per_node] * inference_nodes,
            use_gpus                    = True,
            num_gpus_per_node           = inference_gpus_per_node,
            max_colocated_worker_groups = 1,
        )
        print(
            f"  ✓ Ray inference cluster initialized with {inference_nodes} nodes "
            f"with {inference_gpus_per_node} GPUs per node",
            flush=True,
        )

    # ==========================
    #   Training and Inference
    # ==========================
    print("\n▶ Setting up model and training...", flush=True)

    generation_config["model_name"] = policy_config["model_name"]
    worker_init_timing_metrics: dict[str, float] = {}

    weights_path, optimizer_path = checkpointer.get_resume_paths(last_checkpoint_path)
    if last_checkpoint_path is None:
        pretrained_cfg = checkpointing_config.get("pretrained_checkpoint")
        if pretrained_cfg is not None:
            weights_path, optimizer_path = checkpointer.get_pretrained_paths(pretrained_cfg)

    init_reference_model = loss_config.reference_policy_kl_penalty > 0

    if not init_reference_model and not grpo_config.get(
        "skip_reference_policy_logprobs_calculation"
    ):
        grpo_config["skip_reference_policy_logprobs_calculation"] = True
        print(
            "Auto-enabling `grpo.skip_reference_policy_logprobs_calculation=True` "
            "because `loss_fn.reference_policy_kl_penalty == 0` "
            "(reference model is not loaded)."
        )

    def init_policy():
        t0 = time.perf_counter()
        common: dict[str, Any] = dict(
            cluster              = train_cluster,
            config               = policy_config,
            tokenizer            = tokenizer,
            processor            = processor,
            weights_path         = weights_path,
            optimizer_path       = optimizer_path,
            init_optimizer       = True,
            init_reference_model = init_reference_model,
        )
        dp_cfg = getattr(master_config, "data_plane", None)
        if dp_cfg and dp_cfg.get("enabled", False):
            # Non-colocated experience transfer: TQPolicy bootstraps the
            # data-plane controller, attaches every worker as a client, and
            # exposes the meta-driven *_from_meta dispatch the async loop uses.
            from dockyard_rl.models.policy.tq_policy import TQPolicy

            p: Policy = TQPolicy(**common, dp_cfg=dp_cfg)
        else:
            p = Policy(**common)
        return p, time.perf_counter() - t0

    def init_vllm():
        t0 = time.perf_counter()
        pg = VllmGeneration(
            cluster     = inference_cluster,
            config      = cast(VllmConfig, generation_config),
            parallelism = master_config.policy.get("inference_parallelism", {}),
        )
        pg.finish_generation()
        return pg, time.perf_counter() - t0

    def init_sglang():
        t0 = time.perf_counter()
        pg = SGLangGeneration(
            cluster = inference_cluster,
            config  = cast(SGLangConfig, generation_config),
        )
        pg.finish_generation()
        return pg, time.perf_counter() - t0

    def initialize_generation_with_policy(
        init_generation_fn,
        generation_name:            str,
        init_time_key:              str,
        colocated_inference:        bool,
        worker_init_timing_metrics: dict,
    ):
        use_parallel_init = not colocated_inference

        if use_parallel_init:
            print(
                "  ⚡ Using parallel worker initialization (non-colocated mode)",
                flush=True,
            )
            parallel_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as executor:
                generation_future = executor.submit(init_generation_fn)
                policy_future     = executor.submit(init_policy)
                policy_generation, generation_time = generation_future.result()
                policy,            policy_time     = policy_future.result()
            parallel_wall_time = time.perf_counter() - parallel_start
            worker_init_timing_metrics[init_time_key]          = generation_time
            worker_init_timing_metrics["policy_init_time_s"]   = policy_time
            worker_init_timing_metrics["parallel_wall_time_s"] = parallel_wall_time
            worker_init_timing_metrics["parallel_init_enabled"] = True
        else:
            print(
                "  ⚙️  Using sequential worker initialization (colocated mode)",
                flush=True,
            )
            policy_generation, generation_time = init_generation_fn()
            worker_init_timing_metrics[init_time_key] = generation_time
            policy, policy_time = init_policy()
            worker_init_timing_metrics["policy_init_time_s"]    = policy_time
            worker_init_timing_metrics["parallel_init_enabled"] = 0.0

        return policy_generation, policy

    backend = generation_config["backend"]

    if backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        if generation_config["vllm_cfg"]["precision"] == "fp8":
            assert loss_config.use_importance_sampling_correction, (
                "Importance sampling must be enabled for vLLM FP8 generation "
                "for good convergence!"
            )
        if generation_config["vllm_cfg"]["kv_cache_dtype"].startswith("fp8"):
            assert generation_config["vllm_cfg"]["precision"] == "fp8", (
                f"kv_cache_dtype='{generation_config['vllm_cfg']['kv_cache_dtype']}' "
                "requires precision='fp8'. "
                "FP8 KV cache can only be used together with FP8 model weights."
            )
            assert policy_config.get("dtensor_cfg", {}).get("enabled", False) == False, (
                "DTensor backend is not supported with kv cache fp8 enabled."
            )
            assert not _should_use_async_rollouts(master_config), (
                "Async rollouts is not supported with kv cache fp8 enabled."
            )

        generation_config["vllm_kwargs"]["hf_overrides"] = policy_config.get(
            "hf_config_overrides", {}
        )

        policy_generation, policy = initialize_generation_with_policy(
            init_generation_fn          = init_vllm,
            generation_name             = "vLLM",
            init_time_key               = "vllm_init_time_s",
            colocated_inference         = colocated_inference,
            worker_init_timing_metrics  = worker_init_timing_metrics,
        )
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}",
            flush=True,
        )
    elif backend == "sglang":
        generation_config = cast(SGLangConfig, generation_config)
        generation_config["sglang_cfg"].setdefault(
            "model_path", policy_config["model_name"]
        )

        policy_generation, policy = initialize_generation_with_policy(
            init_generation_fn          = init_sglang,
            generation_name             = "SGLang",
            init_time_key               = "sglang_init_time_s",
            colocated_inference         = colocated_inference,
            worker_init_timing_metrics  = worker_init_timing_metrics,
        )
        print(
            f"  ✓ Using SGLang backend for generation with {policy_config['model_name']}",
            flush=True,
        )
    else:
        raise ValueError(
            f"Unsupported generation backend {backend!r}. "
            "Dockyard supports 'vllm' and 'sglang'."
        )

    worker_init_complete_time = time.perf_counter() - setup_start_time
    policy.print_node_ip_and_gpu_id()

    t0 = time.perf_counter()
    weight_synchronizer = create_weight_synchronizer(
        policy=policy,
        generation=policy_generation,
        generation_backend=backend,
        colocated=colocated_inference,
        train_cluster=train_cluster,
        inference_world_size=inference_nodes * inference_gpus_per_node,
        refit_buffer_size_gb=None,
    )
    weight_synchronizer.init_communicator()
    if not colocated_inference:
        worker_init_timing_metrics["collective_init_time_s"] = (
            time.perf_counter() - t0
        )

    total_setup_time = time.perf_counter() - setup_start_time
    worker_init_timing_metrics["total_setup_time_s"] = total_setup_time

    if worker_init_timing_metrics:
        print("\n▶ Worker Initialization Timing:")
        vllm_time   = worker_init_timing_metrics.get("vllm_init_time_s", 0)
        policy_time = worker_init_timing_metrics.get("policy_init_time_s", 0)
        total_setup = worker_init_timing_metrics.get("total_setup_time_s", 0)
        if vllm_time:
            print(f"  vLLM init: {vllm_time:.1f}s")
        if policy_time:
            print(f"  Policy init: {policy_time:.1f}s")
        other_time = total_setup - worker_init_complete_time
        worker_init_timing_metrics["other_setup_time_s"] = other_time
        print(f"  Other setup: {other_time:.1f}s")
        print(f"  Total setup: {total_setup:.1f}s")
        logger.log_metrics(worker_init_timing_metrics, step=0, prefix="timing/setup")

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print(f"  Total setup time: {total_setup_time:.1f}s")
    print("=" * 60 + "\n", flush=True)

    return (
        policy,
        policy_generation,
        (train_cluster, inference_cluster),
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_save_state,
        master_config,
        weight_synchronizer,
    )

# ============================================================
# Core Algorithm Functions
# ============================================================

def dynamic_sampling(
    repeated_batch,
    std:                             torch.Tensor,
    baseline:                        torch.Tensor,
    dynamic_sampling_num_gen_batches: int,
    master_config:                   MasterConfig,
    timer,
    batch_cache = None,
) -> tuple:
    """DAPO-style dynamic sampling — retain only prompts with non-zero std.

    Returns:
        (batch_to_return, is_batch_complete, batch_cache, dynamic_sampling_metrics)
    """
    is_batch_complete = True
    train_prompts_size = (
        master_config.grpo["num_prompts_per_step"]
        * master_config.grpo["num_generations_per_prompt"]
    )
    repeated_batch["baseline"] = baseline
    repeated_batch["std"]      = std
    total_rewards              = repeated_batch["total_reward"]
    dynamic_sampling_metrics: dict = {}

    filtered_repeated_batch = repeated_batch

    if master_config.grpo["use_dynamic_sampling"]:
        with timer.time("dynamic_sampling"):
            non_zero_std_mask = std != 0.0
            keep_prompt_indices = torch.arange(
                len(non_zero_std_mask), device=std.device
            )[non_zero_std_mask].tolist()

            filtered_repeated_batch = repeated_batch.select_indices(keep_prompt_indices)
            filtered_repeated_batch["std"]      = std[keep_prompt_indices]
            filtered_repeated_batch["baseline"] = baseline[keep_prompt_indices]

            filtered_rewards = filtered_repeated_batch["total_reward"]
            filtered_repeated_batch["total_reward"]    = total_rewards
            filtered_repeated_batch["filtered_reward"] = filtered_rewards

            if filtered_repeated_batch.size > 0:
                batch_cache = (
                    filtered_repeated_batch
                    if batch_cache is None
                    else BatchedDataDict.from_batches(
                        [batch_cache, filtered_repeated_batch]
                    )
                )
                filtered_repeated_batch = batch_cache

            filtered_prompts_size = filtered_repeated_batch.size
            print(
                f"Detected {filtered_prompts_size} prompts with non-zero std; "
                f"{train_prompts_size} are required and used for training."
            )

            if filtered_prompts_size < train_prompts_size:
                dynamic_sampling_max_gen_batches = master_config.grpo.get(
                    "dynamic_sampling_max_gen_batches", 10
                )
                assert dynamic_sampling_max_gen_batches > 0, (
                    "When using grpo.use_dynamic_sampling, "
                    "grpo.dynamic_sampling_max_gen_batches must be > 0"
                )
                if dynamic_sampling_num_gen_batches <= dynamic_sampling_max_gen_batches:
                    print(
                        f"Generation sample buffer size: {filtered_prompts_size} is "
                        f"smaller than train_prompts_size: {train_prompts_size}. "
                        f"Processed {dynamic_sampling_num_gen_batches} batches so far "
                        f"out of {dynamic_sampling_max_gen_batches}."
                    )
                    is_batch_complete = False
                else:
                    raise ValueError(
                        f"Dynamic sampling has reached the maximum allowed number of "
                        f"batches ({dynamic_sampling_max_gen_batches}). Consider "
                        "evaluating the complexity of your data or adjusting "
                        "num_prompts_per_step or num_generations_per_prompt."
                    )
            else:
                num_discarded = filtered_prompts_size - train_prompts_size
                dynamic_sampling_metrics[
                    "dynamic_sampling_num_discarded_valid_samples"
                ] = num_discarded
                filtered_repeated_batch = filtered_repeated_batch.slice(
                    0, train_prompts_size
                )

    batch_to_return = (
        filtered_repeated_batch
        if master_config.grpo["use_dynamic_sampling"]
        else repeated_batch
    )
    return batch_to_return, is_batch_complete, batch_cache, dynamic_sampling_metrics

def scale_rewards(repeated_batch, reward_scaling_cfg: RewardScalingConfig):
    """Linearly scale rewards from source range to target range."""
    if not reward_scaling_cfg["enabled"]:
        return repeated_batch

    rewards    = repeated_batch["total_reward"]
    source_min = float(reward_scaling_cfg.get("source_min", 0.0))
    source_max = float(reward_scaling_cfg.get("source_max", 1.0))
    target_min = float(reward_scaling_cfg.get("target_min", 0.0))
    target_max = float(reward_scaling_cfg.get("target_max", 1.0))

    out_of_range_mask = (rewards < source_min) | (rewards > source_max)
    if torch.any(out_of_range_mask):
        print(
            f"[reward_scaling] WARNING: {int(out_of_range_mask.sum())} rewards "
            f"are outside the configured source range [{source_min}, {source_max}]. "
            "Values will be clipped before scaling."
        )

    def _scale(r: torch.Tensor) -> torch.Tensor:
        c = torch.clamp(r, min=source_min, max=source_max)
        return target_min + (c - source_min) / (source_max - source_min) * (
            target_max - target_min
        )

    repeated_batch["total_reward"] = _scale(rewards)
    for key in get_gdpo_reward_component_keys(repeated_batch):
        repeated_batch[key] = _scale(repeated_batch[key])

    return repeated_batch

def _should_use_async_rollouts(master_config: MasterConfig) -> bool:
    """Return True if the configured backend's async engine is enabled."""
    generation_config = master_config.policy.get("generation", {})
    if not generation_config:
        return False
    backend = generation_config.get("backend", "")
    if backend == "vllm":
        return generation_config.get("vllm_cfg", {}).get("async_engine", False)
    if backend == "sglang":
        return generation_config.get("sglang_cfg", {}).get("async_engine", False)
    return False

def _select_async_rollout_fn(master_config: MasterConfig):
    """Return the async rollout function for this run.

    ``grpo.cua_rollout: true`` selects the multimodal computer-use rollout
    (experience/cua/rollout.py), which threads per-turn screenshots into the
    message log; otherwise the standard text multi-turn rollout is used. Lazy
    import keeps the CUA path's optional image deps off the default code path.
    """
    if master_config.grpo.get("cua_rollout", False):
        from dockyard_rl.experience.cua.rollout import run_async_cua_rollout

        return run_async_cua_rollout
    return run_async_multi_turn_rollout

def resolve_invalid_action_cfg(
    master_config: MasterConfig,
) -> Optional[InvalidActionPenaltyConfig]:
    """Return the enabled invalid-action penalty config, or None.

    The text rollouts collect per-turn verdicts when this is non-None; the
    CUA rollout does not, so the combination is rejected rather than silently
    training without the configured penalty.
    """
    cfg = master_config.grpo.get("invalid_action_penalty")
    if cfg is None or not cfg.get("enabled", False):
        return None
    if master_config.grpo.get("cua_rollout", False):
        raise NotImplementedError(
            "grpo.invalid_action_penalty is not supported with "
            "grpo.cua_rollout=true; the CUA rollout does not collect "
            "invalid-action verdicts. Disable one of them."
        )
    return cfg

def resolve_structured_tool_use_cfg(
    master_config: MasterConfig,
) -> Optional[StructuredToolUseConfig]:
    """Return the enabled structured tool-use config, or None.

    Threaded into the text rollouts when non-None (turn-envelope construction,
    decode fidelity, thinking style). The CUA rollout is a separate action
    space that does not implement the structured protocol, so the combination
    is rejected rather than silently ignoring the configured protocol.
    """
    cfg = resolve_structured_tool_use(master_config.grpo.get("structured_tool_use"))
    if cfg is None:
        return None
    if master_config.grpo.get("cua_rollout", False):
        raise NotImplementedError(
            "grpo.structured_tool_use is not supported with "
            "grpo.cua_rollout=true; the CUA rollout uses a separate action "
            "space and does not implement the structured protocol. "
            "Disable one of them."
        )
    return cfg

def _create_advantage_estimator(master_config: MasterConfig):
    """Instantiate the configured advantage estimator."""
    grpo_config = master_config.grpo
    loss_config = master_config.loss_fn

    adv_estimator_config = grpo_config.get(
        "adv_estimator",
        {
            "name":                       "grpo",
            "normalize_rewards":          grpo_config.get("normalize_rewards", True),
            "use_leave_one_out_baseline": grpo_config.get("use_leave_one_out_baseline", False),
            "minus_baseline":             True,
        },
    )

    name = adv_estimator_config["name"]
    _adv_cfg: dict = dict(adv_estimator_config)
    if name == "gdpo":
        adv = GDPOAdvantageEstimator(_adv_cfg, loss_config)
        print("  ✓ Using GDPO advantage estimator (multi-reward)")
    elif name == "grpo":
        adv = GRPOAdvantageEstimator(_adv_cfg, loss_config)
        print("  ✓ Using GRPO advantage estimator")
    elif name == "reinforce_plus_plus":
        adv = ReinforcePlusPlusAdvantageEstimator(_adv_cfg, loss_config)
        print("  ✓ Using Reinforce++ advantage estimator")
    else:
        raise ValueError(f"Invalid adv_estimator name: {name!r}")
    return adv

def extract_initial_prompt_messages(
    message_logs: list,
    original_prompt_lengths: torch.Tensor,
) -> list:
    """Extract the original prompt messages from each log using token length.

    Correctly identifies the original prompt even when it contains assistant
    messages (multi-turn conversation history): walks each log's messages until
    the cumulative token count reaches the recorded prompt length, rather than
    selecting by role.

    Args:
        message_logs: Per-sample list of message logs.
        original_prompt_lengths: Per-sample original prompt token lengths.
    """
    initial_prompt_message_logs = []
    for i, message_log in enumerate(message_logs):
        initial_prompt_log = []
        cumulative_length = 0
        target_length = int(original_prompt_lengths[i].item())
        for message in message_log:
            if cumulative_length >= target_length:
                break
            initial_prompt_log.append(message)
            cumulative_length += len(message["token_ids"])
        initial_prompt_message_logs.append(initial_prompt_log)
    return initial_prompt_message_logs


def add_grpo_token_loss_masks_and_generation_logprobs(message_logs: list) -> None:
    """Add GRPO token loss masks and ensure generation_logprobs in each message.

    Assistant messages can be part of the original multi-turn prompt history.
    Only rollout-generated assistant messages carry generation_logprobs, so that
    field — not the role alone — marks the trainable tokens. Mutates each message
    in place: sets token_loss_mask, and fills a zero generation_logprobs when
    absent.
    """
    for message_log in message_logs:
        for message in message_log:
            token_ids = message["token_ids"]
            if message["role"] == "assistant" and "generation_logprobs" in message:
                message["token_loss_mask"] = torch.ones_like(token_ids)
            else:
                message["token_loss_mask"] = torch.zeros_like(token_ids)
            if "generation_logprobs" not in message:
                message["generation_logprobs"] = torch.zeros_like(
                    token_ids, dtype=torch.float32
                )

def aggregate_rollout_metrics(per_group_metrics: dict[str, list]) -> dict[str, Any]:
    """Aggregate rollout metrics across trajectory groups by metric semantics.

    - non-numeric values: passed through unchanged
    - keys ending "/min" or starting "min_" (but not "_rate"): minimum
    - keys ending "/max" or starting "max_" (but not "_rate"): maximum
    - "total_turns": summed
    - all other numeric metrics: mean
    """
    aggregated: dict[str, Any] = {}
    for k, v in per_group_metrics.items():
        if not isinstance(v[0], (int, float)):
            aggregated[k] = v
        elif k.endswith("/min") or (k.startswith("min_") and not k.endswith("_rate")):
            aggregated[k] = min(v)
        elif k.endswith("/max") or (k.startswith("max_") and not k.endswith("_rate")):
            aggregated[k] = max(v)
        elif k == "total_turns":
            aggregated[k] = sum(v)
        else:
            aggregated[k] = sum(v) / len(v)
    return aggregated

def _log_mixed_rewards_and_advantages_information(
    logger,
    total_steps: int,
    metrics: dict,
    baseline: torch.Tensor,
    advantages: torch.Tensor,
) -> None:
    logger.log_histogram(
        baseline.numpy(), total_steps + 1, "train/baseline_reward/histogram"
    )
    metrics["baseline_reward/pct_0"]     = 100 * (baseline == 0).float().mean().item()
    metrics["baseline_reward/pct_1"]     = 100 * (baseline == 1).float().mean().item()
    metrics["baseline_reward/pct_mixed"] = (
        100 - metrics["baseline_reward/pct_0"] - metrics["baseline_reward/pct_1"]
    )
    logger.log_histogram(
        advantages.numpy(), total_steps + 1, "train/advantages/histogram"
    )
    metrics["advantages/sum"]  = advantages.float().sum().item()
    metrics["advantages/mean"] = advantages.float().mean().item()

def _clip_grpo_advantages(advantages: torch.Tensor, grpo_config) -> torch.Tensor:
    """Clamp normalized advantages to the configured [low, high] bounds."""
    clip_low  = grpo_config.get("advantage_clip_low", None)
    clip_high = grpo_config.get("advantage_clip_high", None)
    if clip_low is not None:
        advantages = advantages.clamp(min=clip_low)
    if clip_high is not None:
        advantages = advantages.clamp(max=clip_high)
    return advantages

def compute_and_apply_seq_logprob_error_masking(
    train_data,
    rewards: torch.Tensor,
    seq_logprob_error_threshold: Optional[float],
) -> tuple[float, int, float]:
    """Compute sequence-level logprob error and optionally mask high-error sequences."""
    token_mask          = train_data["token_mask"][:, 1:]
    sample_mask         = train_data["sample_mask"]
    prev_logprobs       = train_data["prev_logprobs"][:, 1:]
    generation_logprobs = train_data["generation_logprobs"][:, 1:]
    lp_error            = torch.abs(generation_logprobs - prev_logprobs)
    mask                = token_mask * sample_mask.unsqueeze(-1)

    seq_mult_prob_error = (
        torch.exp(lp_error * mask) * mask
    ).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    max_seq_mult_prob_error = (
        seq_mult_prob_error.max().item()
        if seq_mult_prob_error.numel() > 0
        else 0.0
    )

    num_masked_seqs   = 0
    masked_correct_pct = 0.0

    if seq_logprob_error_threshold is not None:
        print(
            f"▶ Applying sequence-level logprob error masking "
            f"(threshold={seq_logprob_error_threshold})...",
            flush=True,
        )
        original_sample_mask = sample_mask.clone()
        seq_error_mask = (
            seq_mult_prob_error <= seq_logprob_error_threshold
        ).float() * original_sample_mask

        diff_mask      = original_sample_mask - seq_error_mask
        num_masked_seqs = int(diff_mask.sum().item())

        masked_correct_count = 0
        if num_masked_seqs > 0:
            diff_mask_bool        = diff_mask.bool()
            masked_correct_count  = (
                rewards.view(-1)[diff_mask_bool] == 1
            ).sum().item()
            masked_correct_pct = masked_correct_count / num_masked_seqs

        train_data["sample_mask"] = seq_error_mask

        print(
            f"  Masked {num_masked_seqs} sequences with "
            f"mult_prob_error > {seq_logprob_error_threshold}",
            flush=True,
        )
        if num_masked_seqs > 0:
            print(
                f"  • {masked_correct_count}/{num_masked_seqs} masked sequences "
                f"were correct (reward=1) → {masked_correct_pct:.2%}",
                flush=True,
            )

    return max_seq_mult_prob_error, num_masked_seqs, masked_correct_pct

def _dataplane_train_step(
    policy,
    step_meta,
    driver_carry,
    *,
    master_config: MasterConfig,
    adv_estimator,
    loss_fn: LossFunction,
    weight_synchronizer: WeightSynchronizer,
    timer,
) -> tuple[dict[str, Any], float, int, float, dict[str, Any]]:
    """Meta-driven counterpart of the async consume body for the data plane.

    Mirrors the colocated flow exactly, but bulk tensors stay on the plane:
    workers compute per-token logprobs and write them back; the driver pulls
    only the per-token columns it needs for masking / advantage, writes the
    advantage + post-masking sample_mask deltas, then dispatches the
    meta-driven train. Returns the same locals the post-train logging consumes.
    """
    if master_config.loss_fn.force_on_policy_ratio:
        # Skipping prev_logprobs leaves the train partition without the
        # prev_logprobs column train_presharded fetches; unsupported here.
        raise NotImplementedError(
            "data_plane async path does not support "
            "loss_fn.force_on_policy_ratio=True; disable one of them."
        )

    # Per-sample protocol penalty on the carry rewards; counts travel in
    # driver_carry, so this is identical arithmetic to the colocated path.
    driver_carry = apply_invalid_action_penalty(
        driver_carry, master_config.grpo.get("invalid_action_penalty")
    )
    rewards            = driver_carry["total_reward"]
    prompt_ids_for_adv = driver_carry["prompt_ids_for_adv"]
    compute_ref = not master_config.grpo.get(
        "skip_reference_policy_logprobs_calculation"
    )

    # Workers compute per-token logprobs and commit them to the plane.
    print("▶ Preparing for logprob inference...")
    with timer.time("logprob_inference_prep"):
        policy.prepare_for_lp_inference()
    print("▶ Computing logprobs (meta-driven)...", flush=True)
    with timer.time("policy_and_reference_logprobs"):
        policy.get_logprobs_from_meta(step_meta, timer=timer)
        if compute_ref:
            policy.get_reference_policy_logprobs_from_meta(step_meta, timer=timer)

    # Driver pulls only the per-token columns needed for masking / advantage;
    # bulk (input_ids, multimodal) stays on the plane for train_presharded.
    select_fields = ["prev_logprobs", "generation_logprobs", "token_mask", "sample_mask"]
    if compute_ref:
        select_fields.append("reference_policy_logprobs")
    extras = policy.read_from_dataplane(step_meta, select_fields=select_fields)
    if not compute_ref:
        extras["reference_policy_logprobs"] = torch.zeros_like(extras["prev_logprobs"])

    seq_logprob_error_threshold = master_config.grpo.get(
        "seq_logprob_error_threshold", None
    )
    (
        max_seq_mult_prob_error,
        num_masked_seqs,
        masked_correct_pct,
    ) = compute_and_apply_seq_logprob_error_masking(
        train_data=extras,
        rewards=rewards,
        seq_logprob_error_threshold=seq_logprob_error_threshold,
    )

    with timer.time("advantage_calculation"):
        print("▶ Computing advantages...", flush=True)
        token_mask  = extras["token_mask"]
        sample_mask = extras["sample_mask"]
        mask        = token_mask * sample_mask.unsqueeze(-1)

        # GRPO / Reinforce++ ignore repeated_batch; GDPO reads per-component
        # reward keys from the carry-derived adv_inputs (same payload as
        # passing the full batch).
        adv_inputs = BatchedDataDict[Any]({"total_reward": rewards})
        for k in get_gdpo_reward_component_keys(driver_carry):
            adv_inputs[k] = driver_carry[k]
        advantages = adv_estimator.compute_advantage(
            prompt_ids         = prompt_ids_for_adv,
            rewards            = rewards,
            mask               = mask,
            repeated_batch     = adv_inputs,
            logprobs_policy    = extras["prev_logprobs"],
            logprobs_reference = extras.get("reference_policy_logprobs"),
        )
        advantages = _clip_grpo_advantages(advantages, master_config.grpo)

    # Driver delta-write: advantages + post-masking sample_mask under the same
    # sample ids so train_presharded fetches the union. When the reference
    # logprobs were skipped, write zeros so the train fetch's schema is complete.
    write_fields: dict[str, Any] = {
        "advantages": advantages,
        "sample_mask": sample_mask,
    }
    if not compute_ref:
        write_fields["reference_policy_logprobs"] = extras[
            "reference_policy_logprobs"
        ]
    policy.write_to_dataplane(step_meta, fields=write_fields)

    print("▶ Preparing for training...")
    with timer.time("training_prep"):
        policy.prepare_for_training()
        weight_synchronizer.mark_stale()

    print("▶ Training policy (meta-driven)...")
    with timer.time("policy_training"):
        train_results = policy.train_from_meta(step_meta, loss_fn=loss_fn, timer=timer)

    # Per-token + per-row values for post-train metrics / slim logging. The
    # driver already holds these from the read-back above (no extra fetch);
    # input_ids stays on the plane, so DP-mode jsonl omits the token bulk.
    dp_payload: dict[str, Any] = {
        "rewards": rewards,
        "advantages": advantages,
        "token_mask": extras["token_mask"],
        "sample_mask": sample_mask,
        "generation_logprobs": extras["generation_logprobs"],
        "prev_logprobs": extras["prev_logprobs"],
        "input_lengths": driver_carry["input_lengths"],
        "length": driver_carry["length"],
    }
    return (
        train_results,
        max_seq_mult_prob_error,
        num_masked_seqs,
        masked_correct_pct,
        dp_payload,
    )

# ============================================================
# Training & Validation
# ============================================================

def grpo_train(
    policy,
    policy_generation: Optional[GenerationInterface],
    wrapped_dataloader,
    val_dataloader,
    tokenizer:       PreTrainedTokenizerBase,
    loss_fn:         LossFunction,
    task_to_env:     dict,
    val_task_to_env: Optional[dict],
    logger,
    checkpointer,
    grpo_save_state: GRPOSaveState,
    master_config:   MasterConfig,
    weight_synchronizer: WeightSynchronizer,
) -> None:
    """Run synchronous GRPO training algorithm."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout            = master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time = True,
    )
    timeout.start_iterations()
    memory_tracker = MemoryTracker()

    kv_scales_cache = None

    NEED_REFIT = True
    if policy_generation is None:
        policy_generation = policy
        NEED_REFIT = False
    weight_synchronizer.mark_stale()
    assert policy_generation is not None

    sync_kv_scales = getattr(policy_generation, "requires_kv_scale_sync", False)

    current_step     = grpo_save_state["current_step"]
    total_steps      = grpo_save_state["total_steps"]
    max_num_steps    = master_config.grpo["max_num_steps"]
    current_epoch    = grpo_save_state["current_epoch"]
    max_num_epochs   = master_config.grpo["max_num_epochs"]
    consumed_samples = grpo_save_state["consumed_samples"]
    total_valid_tokens = grpo_save_state.get("total_valid_tokens", 0)
    val_at_start     = master_config.grpo["val_at_start"]
    val_at_end       = master_config.grpo["val_at_end"]
    val_period       = master_config.grpo["val_period"]
    colocated_inference = master_config.policy["generation"]["colocated"]["enabled"]

    adv_estimator = _create_advantage_estimator(master_config)

    if val_at_start and current_step == 0:
        print("\n🔍 Running initial validation...", flush=True)
        memory_tracker.snapshot_start_of_stage("Initial validation", dir())
        if NEED_REFIT and weight_synchronizer.is_stale:
            weight_synchronizer.sync_weights()
        else:
            policy_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=master_config,
            logger=logger,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, current_step, prefix="validation")
        logger.log_metrics(validation_timings, current_step, prefix="timing/validation")

    if master_config.data["use_multiple_dataloader"]:
        warnings.warn(
            "When using multiple dataloaders, MultipleDataloaderWrapper operates "
            "as an infinite iterator. As a result, grpo.max_num_epochs will be "
            "ignored, and only grpo.max_num_steps will be used."
        )

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        memory_tracker.snapshot_start_of_stage("Preparing batch", dir())
        print(
            f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}"
        )
        batch_cache: Any = None
        dynamic_sampling_num_gen_batches = 0

        for batch in wrapped_dataloader:
            metrics_logging_data: dict = {}
            metrics: dict             = {}

            if master_config.data["use_multiple_dataloader"]:
                print(
                    f"\n{'=' * 25} Step {current_step + 1}/{max_num_steps} {'=' * 25}",
                    flush=True,
                )
            else:
                print(
                    f"\n{'=' * 25} Step {current_step + 1}/"
                    f"{min(len(wrapped_dataloader), max_num_steps)} {'=' * 25}",
                    flush=True,
                )

            maybe_gpu_profile_step(policy, total_steps + 1)
            if policy != policy_generation and hasattr(policy_generation, "start_gpu_profiling"):
                maybe_gpu_profile_step(cast(Any, policy_generation), total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # Prepare batch
                print("▶ Preparing batch...", flush=True)
                with timer.time("data_processing"):
                    repeated_batch = batch.repeat_interleave(
                        master_config.grpo["num_generations_per_prompt"]
                    )
                    assert tokenizer.pad_token_id is not None, (
                        "tokenizer.pad_token_id must be set before generation"
                    )
                    batched_flat, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict=cast(dict[str, int], {"token_ids": tokenizer.pad_token_id}),
                    )
                    input_ids = batched_flat["token_ids"]

                # Generation
                memory_tracker.snapshot_start_of_stage("Generation", dir())
                print(
                    f"▶ Generating responses for batch of size {repeated_batch.size}...",
                    flush=True,
                )
                with timer.time("prepare_for_generation/total"):
                    if NEED_REFIT and weight_synchronizer.is_stale:
                        if sync_kv_scales and kv_scales_cache is None:
                            print("▶ Computing KV cache scales...", flush=True)
                            policy.prepare_for_lp_inference()
                            calib_flat, calib_input_lengths = (
                                batched_message_log_to_flat_message(
                                    repeated_batch["message_log"],
                                    pad_value_dict=cast(dict[str, int], {"token_ids": tokenizer.pad_token_id}),
                                    make_sequence_length_divisible_by=master_config.policy[
                                        "make_sequence_length_divisible_by"
                                    ],
                                )
                            )
                            calibration_data = BatchedDataDict[ClippedPGLossDataDict](
                                {
                                    "input_ids":     calib_flat["token_ids"],
                                    "input_lengths": calib_input_lengths,
                                }
                            )
                            calibration_data.update(
                                calib_flat.get_multimodal_dict(as_tensors=False)
                            )
                            calibration_data.to("cpu")
                            kv_scales_cache = policy.calibrate_qkv_fp8_scales(
                                calibration_data, include_q=True
                            )["layers"]

                        weight_synchronizer.sync_weights(
                            timer=timer,
                            kv_scales=kv_scales_cache if sync_kv_scales else None,
                        )
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()
                        policy_generation.prepare_for_generation()

                dynamic_sampling_num_gen_batches += 1
                if dynamic_sampling_num_gen_batches == 1 and hasattr(
                    policy_generation, "snapshot_step_metrics"
                ):
                    policy_generation.snapshot_step_metrics()

                generation_logger_metrics = None
                with timer.time("generation"):
                    if policy_generation is not None:
                        policy_generation.clear_logger_metrics()

                    # Non-None only for the text rollouts (resolvers reject
                    # the CUA combination), so the kwargs are always accepted.
                    invalid_action_cfg = resolve_invalid_action_cfg(master_config)
                    structured_cfg = resolve_structured_tool_use_cfg(master_config)
                    rollout_extra_kwargs: dict[str, Any] = {}
                    if invalid_action_cfg is not None:
                        rollout_extra_kwargs["invalid_action_cfg"] = invalid_action_cfg
                    if structured_cfg is not None:
                        rollout_extra_kwargs["structured_cfg"] = structured_cfg
                    if _should_use_async_rollouts(master_config):
                        repeated_batch, rollout_metrics = _select_async_rollout_fn(master_config)(
                            policy_generation = policy_generation,
                            input_batch       = repeated_batch,
                            tokenizer         = tokenizer,
                            task_to_env       = task_to_env,
                            max_seq_len       = master_config.policy[
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns = master_config.grpo["max_rollout_turns"],
                            greedy            = False,
                            **rollout_extra_kwargs,
                        )
                    else:
                        repeated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation = policy_generation,
                            input_batch       = repeated_batch,
                            tokenizer         = tokenizer,
                            task_to_env       = task_to_env,
                            max_seq_len       = master_config.policy[
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns = master_config.grpo["max_rollout_turns"],
                            greedy            = False,
                            **rollout_extra_kwargs,
                        )

                    policy_generation.finish_generation()
                    if policy_generation is not None:
                        generation_logger_metrics = (
                            policy_generation.get_logger_metrics()
                        )

                    metrics_logging_data["mean_gen_tokens_per_sample"] = (
                        rollout_metrics["mean_gen_tokens_per_sample"]
                    )
                    logger.log_metrics(rollout_metrics, total_steps + 1, prefix="train")

                repeated_batch = scale_rewards(
                    repeated_batch, master_config.grpo["reward_scaling"]
                )
                if master_config.grpo["reward_shaping"]["enabled"]:
                    repeated_batch = apply_reward_shaping(
                        repeated_batch, master_config.grpo["reward_shaping"]
                    )
                repeated_batch = apply_invalid_action_penalty(
                    repeated_batch,
                    master_config.grpo.get("invalid_action_penalty"),
                    step=total_steps,
                )

                # Rewards & advantages
                memory_tracker.snapshot_start_of_stage("Processing rewards", dir())
                print("▶ Processing rewards...,", flush=True)
                with timer.time("reward_calculation"):
                    rewards = repeated_batch["total_reward"]

                    # For DAPO with reward shaping, compute std on the raw
                    # pre-shaping reward so dynamic sampling filters on the task
                    # metric, not length-dependent shaped variance. Baseline
                    # (which drives advantages) stays on the shaped reward.
                    std_rewards = (
                        repeated_batch["unshaped_total_reward"]
                        if master_config.grpo["use_dynamic_sampling"]
                        and "unshaped_total_reward" in repeated_batch
                        else None
                    )

                    print("▶ Computing advantages...", flush=True)
                    if master_config.grpo.get("calculate_advantages_on_gpu"):
                        print("Computing advantages on GPU!")
                        device_id = 0
                        baseline, std = calculate_baseline_and_std_per_prompt(
                            input_ids.cuda(device_id),
                            rewards.cuda(device_id),
                            torch.ones_like(rewards).cuda(device_id),
                            leave_one_out_baseline=master_config.grpo[
                                "use_leave_one_out_baseline"
                            ],
                            std_rewards=(
                                std_rewards.cuda(device_id)
                                if std_rewards is not None
                                else None
                            ),
                        )
                        baseline = baseline.cpu()
                        std      = std.cpu()
                    else:
                        baseline, std = calculate_baseline_and_std_per_prompt(
                            input_ids,
                            rewards,
                            torch.ones_like(rewards),
                            leave_one_out_baseline=master_config.grpo[
                                "use_leave_one_out_baseline"
                            ],
                            std_rewards=std_rewards,
                        )

                    repeated_batch, is_batch_complete, batch_cache, ds_metrics = (
                        dynamic_sampling(
                            repeated_batch,
                            std,
                            baseline,
                            dynamic_sampling_num_gen_batches,
                            master_config,
                            timer,
                            batch_cache,
                        )
                    )
                    if ds_metrics:
                        ds_metrics["dynamic_sampling_num_gen_batches"] = (
                            dynamic_sampling_num_gen_batches
                        )

                    rewards  = (
                        repeated_batch["total_reward"]
                        if not master_config.grpo["use_dynamic_sampling"]
                        else repeated_batch["filtered_reward"]
                    )
                    baseline = repeated_batch["baseline"]
                    std      = repeated_batch["std"]

                    if not is_batch_complete:
                        continue

                    gen_step_metrics: dict = {}
                    if hasattr(policy_generation, "get_step_metrics"):
                        gen_step_metrics = policy_generation.get_step_metrics()

                    baseline_for_log = baseline.clone()

                    initial_prompt_message_logs = extract_initial_prompt_messages(
                        repeated_batch["message_log"],
                        repeated_batch["length"],
                    )
                    prompt_batched_flat, _ = batched_message_log_to_flat_message(
                        initial_prompt_message_logs,
                        pad_value_dict=cast(dict[str, int], {"token_ids": tokenizer.pad_token_id}),
                    )
                    prompt_ids_for_adv = prompt_batched_flat["token_ids"]
                    del initial_prompt_message_logs
                    del prompt_batched_flat
                    del input_ids
                    del baseline
                    del std

                # Training data preparation
                with timer.time("data_processing"):
                    use_overlong_filtering = master_config.grpo.get(
                        "overlong_filtering", False
                    )
                    if use_overlong_filtering:
                        loss_multiplier = repeated_batch["loss_multiplier"].clone()
                        truncated       = repeated_batch["truncated"]
                        if isinstance(truncated, list):
                            truncated = torch.tensor(truncated, dtype=torch.bool)
                        loss_multiplier[truncated] = 0
                        repeated_batch["loss_multiplier"] = loss_multiplier

                    add_grpo_token_loss_masks_and_generation_logprobs(
                        repeated_batch["message_log"]
                    )

                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict=cast(dict[str, int], {"token_ids": tokenizer.pad_token_id}),
                        make_sequence_length_divisible_by=master_config.policy[
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    train_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids":           flat_messages["token_ids"],
                            "input_lengths":       input_lengths,
                            "generation_logprobs": flat_messages["generation_logprobs"],
                            "token_mask":          flat_messages["token_loss_mask"],
                            "sample_mask":         repeated_batch["loss_multiplier"],
                        }
                    )
                    extra_multimodal_data = flat_messages.get_multimodal_dict(
                        as_tensors=False
                    )
                    train_data.update(extra_multimodal_data)
                    train_data.to("cpu")
                    metrics_logging_data["content"] = flat_messages["content"]

                # Logprob inference
                memory_tracker.snapshot_start_of_stage("Computing logprobs", dir())
                skip_prev_logprobs = master_config.loss_fn.force_on_policy_ratio
                if skip_prev_logprobs:
                    print(
                        "▶ Skipping prev_logprobs (force_on_policy_ratio=True)...",
                        flush=True,
                    )
                else:
                    print("▶ Preparing for logprob inference...", flush=True)
                    with timer.time("logprob_inference_prep"):
                        policy.prepare_for_lp_inference()

                print("▶ Computing logprobs...", flush=True)
                with timer.time("policy_and_reference_logprobs"):
                    logprob_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids":     train_data["input_ids"],
                            "input_lengths": train_data["input_lengths"],
                            "token_mask":    flat_messages["token_loss_mask"],
                            "sample_mask":   repeated_batch["loss_multiplier"],
                            **extra_multimodal_data,
                        }
                    )
                    if not skip_prev_logprobs:
                        train_data["prev_logprobs"] = policy.get_logprobs(
                            logprob_data, timer=timer
                        )["logprobs"]
                    else:
                        train_data["prev_logprobs"] = torch.zeros_like(
                            train_data["generation_logprobs"]
                        )

                    if not master_config.grpo.get(
                        "skip_reference_policy_logprobs_calculation"
                    ):
                        train_data["reference_policy_logprobs"] = (
                            policy.get_reference_policy_logprobs(
                                logprob_data, timer=timer
                            )["reference_logprobs"]
                        )
                    else:
                        train_data["reference_policy_logprobs"] = torch.zeros_like(
                            train_data["prev_logprobs"]
                        )

                    del logprob_data
                    del extra_multimodal_data

                seq_logprob_error_threshold = master_config.grpo.get(
                    "seq_logprob_error_threshold", None
                )
                if skip_prev_logprobs:
                    assert seq_logprob_error_threshold is None, (
                        "seq_logprob_error_threshold requires prev_logprobs; "
                        "cannot use with force_on_policy_ratio=True"
                    )
                    max_seq_mult_prob_error = 0.0
                    num_masked_seqs        = 0
                    masked_correct_pct     = 0.0
                else:
                    (
                        max_seq_mult_prob_error,
                        num_masked_seqs,
                        masked_correct_pct,
                    ) = compute_and_apply_seq_logprob_error_masking(
                        train_data=train_data,
                        rewards=rewards,
                        seq_logprob_error_threshold=seq_logprob_error_threshold,
                    )

                # Advantage computation
                with timer.time("advantage_calculation"):
                    print("▶ Computing advantages...", flush=True)
                    token_mask  = train_data["token_mask"]
                    sample_mask = train_data["sample_mask"]
                    mask        = token_mask * sample_mask.unsqueeze(-1)

                    train_data["advantages"] = adv_estimator.compute_advantage(
                        prompt_ids         = prompt_ids_for_adv,
                        rewards            = rewards,
                        mask               = mask,
                        repeated_batch     = repeated_batch,
                        logprobs_policy    = train_data["prev_logprobs"],
                        logprobs_reference = train_data.get(
                            "reference_policy_logprobs"
                        ),
                    )
                    # Advantage-locus invalid-action penalties on flagged spans (N2).
                    train_data["advantages"], _ = apply_message_span_advantage_penalties(
                        train_data["advantages"],
                        repeated_batch["message_log"],
                        master_config.grpo.get("invalid_action_penalty"),
                        step=total_steps,
                    )
                    train_data["advantages"] = _clip_grpo_advantages(
                        train_data["advantages"], master_config.grpo
                    )
                    del prompt_ids_for_adv

                    _log_mixed_rewards_and_advantages_information(
                        logger       = logger,
                        total_steps  = total_steps,
                        metrics      = metrics,
                        baseline     = baseline_for_log,
                        advantages   = train_data["advantages"],
                    )
                    del baseline_for_log

                # Policy training
                memory_tracker.snapshot_start_of_stage("Policy train", dir())
                print("▶ Preparing for training...", flush=True)
                with timer.time("training_prep"):
                    policy.prepare_for_training()
                    weight_synchronizer.mark_stale()

                print("▶ Training policy...", flush=True)
                with timer.time("policy_training"):
                    train_results = policy.train(
                        train_data,
                        loss_fn,
                        timer=timer,
                    )

                if sync_kv_scales:
                    with timer.time("recompute_kv_scales"):
                        print(
                            "▶ Recomputing KV cache scales after policy update...",
                            flush=True,
                        )
                        kv_scales_cache     = policy.calibrate_qkv_fp8_scales(
                            train_data, include_q=True
                        )["layers"]
                        weight_synchronizer.mark_stale()

                is_last_step = total_steps + 1 >= max_num_steps
                if not master_config.data["use_multiple_dataloader"]:
                    is_last_step = is_last_step or (
                        (current_epoch + 1 == max_num_epochs)
                        and (current_step + 1 == len(wrapped_dataloader))
                    )

                # Validation
                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
                    memory_tracker.snapshot_start_of_stage("Validation", dir())
                    if NEED_REFIT and weight_synchronizer.is_stale:
                        weight_synchronizer.sync_weights(
                            kv_scales=kv_scales_cache if sync_kv_scales else None,
                        )
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()
                        policy_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        policy_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=total_steps + 1,
                        master_config=master_config,
                        logger=logger,
                    )
                    policy_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )

                # Metrics assembly
                flat_advantages  = train_data["advantages"]
                flat_token_mask  = flat_messages["token_loss_mask"]
                response_advantages = torch.masked_select(
                    flat_advantages, flat_token_mask.bool()
                )

                memory_tracker.snapshot_start_of_stage("Metrics", dir())
                metrics = {
                    **metrics,
                    "loss":               train_results["loss"].numpy(),
                    "grad_norm":          train_results["grad_norm"].numpy(),
                    "reward":             rewards.numpy(),
                    "mean_prompt_length": repeated_batch["length"].numpy(),
                    "total_num_tokens":   input_lengths.numpy(),
                    "advantages/mean":    torch.mean(response_advantages).detach().item()
                    if response_advantages.numel() > 0 else 0.0,
                    "advantages/max":     torch.max(response_advantages).detach().item()
                    if response_advantages.numel() > 0 else 0.0,
                    "advantages/min":     torch.min(response_advantages).detach().item()
                    if response_advantages.numel() > 0 else 0.0,
                    **ds_metrics,
                }
                if "moe_metrics" in train_results:
                    metrics.update(
                        {f"moe/{k}": v for k, v in train_results["moe_metrics"].items()}
                    )
                if master_config.grpo["use_dynamic_sampling"]:
                    metrics["filtered_reward"] = rewards.numpy()
                    metrics["reward"]           = repeated_batch["total_reward"].numpy()

                metrics.update(train_results["all_mb_metrics"])
                metrics.update(gen_step_metrics)
                for k, v in metrics.items():
                    if k in {"probs_ratio_min", "probs_ratio_clamped_min"}:
                        valid_values = [x for x in v if not np.isinf(x)]
                        metrics[k]   = (
                            np.min(valid_values).item() if valid_values else -1.0
                        )
                    elif k in {"probs_ratio_max", "probs_ratio_clamped_max"}:
                        valid_values = [x for x in v if not np.isinf(x)]
                        metrics[k]   = (
                            np.max(valid_values).item() if valid_values else -1.0
                        )
                    elif k in {
                        "lr", "wd", "reward", "filtered_reward",
                        "global_valid_seqs", "global_valid_toks",
                        "mean_prompt_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    elif isinstance(v, (np.ndarray, list)):
                        metrics[k] = np.sum(v).item()
                    else:
                        print(f"Skipping aggregation for {k} ({type(v)})")

                metrics.update(rollout_metrics)
                metrics["generation_logger_metrics"]          = generation_logger_metrics
                total_valid_tokens                           += metrics["global_valid_toks"]
                metrics["max_seq_mult_prob_error"]            = max_seq_mult_prob_error
                metrics["num_masked_seqs_by_logprob_error"]  = num_masked_seqs
                metrics["masked_correct_pct"]                 = masked_correct_pct

                # Checkpointing
                consumed_samples += master_config.grpo["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config.checkpointing["save_period"] == 0
                )
                should_save_by_timeout = timeout.check_save()

                memory_tracker.snapshot_start_of_stage("Checkpointing", dir())
                if master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    policy.prepare_for_training()

                    grpo_save_state["current_step"]      = current_step + 1
                    grpo_save_state["total_steps"]       = total_steps + 1
                    grpo_save_state["current_epoch"]     = current_epoch
                    grpo_save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        grpo_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in grpo_save_state:
                        del grpo_save_state["val_reward"]
                    grpo_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config.checkpointing["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with "
                            "'val:' or 'train:', followed by the metric name."
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on "
                                f"{metric_name} but no {prefix} metrics were "
                                "collected. This checkpoint will not be saved "
                                "as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in grpo_save_state:
                                del grpo_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            grpo_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, grpo_save_state, master_config
                        )
                        policy.save_checkpoint(
                            weights_path   = os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path = (
                                os.path.join(
                                    checkpoint_path, "policy", "optimizer"
                                )
                                if checkpointer.save_optimizer
                                else None
                            ),
                            tokenizer_path = os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg = master_config.checkpointing,
                        )
                        if master_config.data["use_multiple_dataloader"]:
                            for (
                                task_name,
                                task_dataloader,
                            ) in wrapped_dataloader.dataloaders.items():
                                torch.save(
                                    task_dataloader.state_dict(),
                                    os.path.join(
                                        checkpoint_path,
                                        f"train_dataloader_{task_name}.pt",
                                    ),
                                )
                        else:
                            torch.save(
                                wrapped_dataloader.state_dict(),
                                os.path.join(checkpoint_path, "train_dataloader.pt"),
                            )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # Logging
            memory_tracker.snapshot_start_of_stage("Logging", dir())
            log_data: dict = {}
            if "agent_ref" in repeated_batch:
                log_data["agent_ref"] = repeated_batch["agent_ref"]
            log_data["content"]  = flat_messages["content"]
            log_data["rewards"]  = rewards.tolist()
            if master_config.grpo["use_dynamic_sampling"]:
                log_data["filtered_rewards"] = rewards.tolist()
                log_data["rewards"]          = repeated_batch["total_reward"].tolist()
            log_data["input_lengths"]      = input_lengths.tolist()
            log_data["token_ids"]          = train_data["input_ids"].tolist()
            log_data["token_loss_mask"]    = train_data["token_mask"].tolist()
            log_data["sample_loss_mask"]   = train_data["sample_mask"].tolist()
            log_data["advantages"]         = train_data["advantages"].tolist()
            log_data["generation_logprobs"] = train_data["generation_logprobs"].tolist()
            log_data["prev_logprobs"]       = train_data["prev_logprobs"].tolist()
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{total_steps + 1}.jsonl"
            )
            del log_data
            del flat_messages

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )

            if metrics["token_mult_prob_error"] > 1.05:
                logger.log_plot_token_mult_prob_error(
                    {
                        "prompt_lengths":    repeated_batch["length"],
                        "full_lengths":      input_lengths,
                        "generation_logprobs": train_data["generation_logprobs"],
                        "prev_logprobs":     train_data["prev_logprobs"],
                        "token_mask":        train_data["token_mask"],
                        "sample_mask":       train_data["sample_mask"],
                    },
                    total_steps + 1,
                    name="train/token_mult_prob_error_plot_sample",
                )
            del train_data

            if (
                master_config.policy["generation"]
                .get("vllm_cfg", {})
                .get("enable_vllm_metrics_logger", False)
                and master_config.logger["wandb_enabled"]
                and generation_logger_metrics is not None
            ):
                log_generation_metrics_to_wandb(
                    generation_logger_metrics,
                    total_steps + 1,
                    master_config.policy["generation"]["vllm_cfg"][
                        "vllm_metrics_logger_interval"
                    ],
                    logger,
                )

            if (
                master_config.policy["generation"]
                .get("vllm_cfg", {})
                .get("async_engine", False)
            ):
                for metric_name in metrics.keys():
                    if metric_name.startswith("histogram/"):
                        logger.log_histogram(
                            metrics[metric_name],
                            total_steps + 1,
                            f"generation_metrics/{metric_name}",
                        )

            print("\n📊 Training Results:")
            print(f"  • Loss: {metrics['loss']:.4f}")
            if "draft_loss" in metrics:
                print(f"  • Draft Loss: {metrics['draft_loss']:.4f}")
            print(f"  • Generation KL Error: {metrics['gen_kl_error']:.4f}")
            if master_config.grpo["use_dynamic_sampling"]:
                print(f"  • Avg Filtered Reward: {np.mean(rewards.numpy()):.4f}")
                print(
                    f"  • Avg Total Reward: "
                    f"{np.mean(repeated_batch['total_reward'].numpy()):.4f}"
                )
            else:
                print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
            print(
                f"  • Mean Generation Length: "
                f"{metrics_logging_data['mean_gen_tokens_per_sample']:.4f}",
                flush=True,
            )

            print("\n⏱️  Timing:", flush=True)
            total_time = timing_metrics.get("total_step_time", 0)
            print(f"  • Total step time: {total_time:.2f}s", flush=True)
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            total_num_gpus = (
                master_config.cluster["num_nodes"]
                * master_config.cluster["gpus_per_node"]
            )
            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            performance_metrics = print_performance_metrics(
                train_results, metrics, timing_metrics, master_config.model_dump()
            )

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(performance_metrics, total_steps + 1, prefix="performance")
            logger.log_metrics(
                timing_metrics,
                total_steps + 1,
                prefix="timing/train",
                step_finished=True,
            )

            batch_cache = None
            dynamic_sampling_num_gen_batches = 0

            memory_tracker.snapshot_start_of_stage("After CPU memory clear", dir())
            del repeated_batch
            del rewards
            del metrics
            if "val_metrics" in dir():
                del val_metrics

            timer.reset()
            current_step += 1
            total_steps  += 1
            if should_save_by_timeout:
                memory_tracker.snapshot_start_of_stage("", dir())
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= max_num_steps:
                memory_tracker.snapshot_start_of_stage("", dir())
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        current_epoch += 1
        current_step   = 0

def validate(
    policy_generation: GenerationInterface,
    val_dataloader,
    tokenizer,
    val_task_to_env,
    step:          int,
    master_config: MasterConfig,
    logger=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        assert master_config.grpo["val_period"] == 0, (
            "val_dataloader is None, so grpo.val_period must be 0"
        )
        print("  ⚠️ No validation dataloader provided, skipping validation", flush=True)
        return {}, {}

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...", flush=True)

        total_rewards:    list[float] = []
        total_lengths:    list[float] = []
        all_message_logs: list        = []

        _max_val_samples = master_config.grpo["max_val_samples"]
        _val_batch_size  = master_config.grpo["val_batch_size"]
        assert _max_val_samples is not None, "max_val_samples must be set when validation is enabled"
        assert _val_batch_size  is not None, "val_batch_size must be set when validation is enabled"
        max_batches = _max_val_samples // _val_batch_size
        additional_metrics_to_report: dict = {}
        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            additional_metrics_to_report = {}

            # Thread the structured protocol into validation rollouts too: the
            # turn-envelope construction + decode fidelity change how the env
            # parses actions, so omitting it would mis-score structured val
            # rollouts. None (default) leaves validation byte-identical.
            structured_cfg = resolve_structured_tool_use_cfg(master_config)
            val_structured_kwargs: dict[str, Any] = (
                {"structured_cfg": structured_cfg}
                if structured_cfg is not None
                else {}
            )
            if _should_use_async_rollouts(master_config):
                val_batch, gen_metrics = _select_async_rollout_fn(master_config)(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len       = master_config.policy["max_total_sequence_length"],
                    max_rollout_turns = master_config.grpo["max_rollout_turns"],
                    greedy            = False,
                    **val_structured_kwargs,
                )
            else:
                val_batch, gen_metrics = run_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len       = master_config.policy["max_total_sequence_length"],
                    max_rollout_turns = master_config.grpo["max_rollout_turns"],
                    greedy            = False,
                    **val_structured_kwargs,
                )

            total_rewards.extend(val_batch["total_reward"].tolist())
            total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

            to_env = [
                get_keys_from_message_log(
                    val_batch["message_log"][i], ["role", "content"]
                )
                for i in range(len(val_batch["message_log"]))
            ]
            all_message_logs.extend(to_env)

        num_samples = len(total_rewards)
        if num_samples > 0:
            rewards_t = torch.tensor(total_rewards, dtype=torch.float32)
            accuracy  = rewards_t.mean().item()
        else:
            accuracy = 0.0

        avg_length = (
            sum(total_lengths) / len(total_lengths) if total_lengths else 0.0
        )

        val_metrics = {
            "accuracy":   accuracy,
            "avg_length": avg_length,
            **additional_metrics_to_report,
        }

        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples = min(
                    master_config.logger.get("num_val_samples_to_print", 5),
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...", flush=True)

    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    print(f"    • Samples processed: {len(total_rewards)}", flush=True)
    print("\n  ⏱️  Validation Timing:")
    print(f"    • Total validation time: {validation_time:.2f}s", flush=True)

    if logger is not None:
        val_log_data = {
            "content": all_message_logs,
            "rewards": total_rewards,
        }
        logger.log_batched_dict_as_jsonl(val_log_data, f"val_data_step{step}.jsonl")

    timer.reset()
    gc.collect()
    torch.cuda.empty_cache()

    return val_metrics, timing_metrics

def async_grpo_train(
    policy,
    policy_generation: Optional[GenerationInterface],
    dataloader,
    val_dataloader,
    tokenizer:                PreTrainedTokenizerBase,
    loss_fn:                  LossFunction,
    task_to_env:              dict,
    val_task_to_env:          Optional[dict],
    logger,
    checkpointer,
    grpo_save_state:          GRPOSaveState,
    master_config:            MasterConfig,
    weight_synchronizer:      WeightSynchronizer,
    max_trajectory_age_steps: int = 1,
) -> None:
    """Run asynchronous GRPO training with replay buffer."""
    assert _should_use_async_rollouts(master_config), (
        "Async GRPO requires a backend with async engine enabled. Set "
        "policy.generation.vllm_cfg.async_engine (vLLM) or "
        "policy.generation.sglang_cfg.async_engine (SGLang) to true."
    )
    assert master_config.loss_fn.use_importance_sampling_correction, (
        "Importance sampling correction must be enabled for async GRPO for "
        "good convergence due to off-policy samples!"
    )

    _async_grpo_cfg = master_config.grpo.get("async_grpo", {})
    if _async_grpo_cfg.get("max_trajectory_age_steps", 1) > 1:
        if not _async_grpo_cfg.get("in_flight_weight_updates", False):
            print(
                "⚠️ WARNING: In-flight weight updates must be enabled for async GRPO "
                "with max_trajectory_age_steps > 1. Without in-flight weight updates, "
                "having more max_trajectory_age_steps will not give any performance benefit."
            )

    timer   = Timer()
    timeout = TimeoutChecker(
        timeout            = master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time = True,
    )
    timeout.start_iterations()
    NEED_REFIT = True

    if policy_generation is None:
        policy_generation = policy
        NEED_REFIT = False
    weight_synchronizer.mark_stale()
    assert policy_generation is not None

    step               = grpo_save_state["current_step"]
    weight_version     = step
    consumed_samples   = grpo_save_state["consumed_samples"]
    total_valid_tokens = grpo_save_state.get("total_valid_tokens", 0)
    val_period         = master_config.grpo["val_period"]
    val_at_start       = master_config.grpo["val_at_start"]
    val_at_end         = master_config.grpo["val_at_end"]
    colocated_inference = master_config.policy["generation"]["colocated"]["enabled"]

    adv_estimator = _create_advantage_estimator(master_config)

    assert not colocated_inference, (
        "Colocated inference is not supported for async GRPO. "
        "Please use non-colocated inference."
    )

    num_prompts_per_step      = master_config.grpo["num_prompts_per_step"]
    samples_per_prompt_group  = master_config.grpo["num_generations_per_prompt"]
    train_gbs                 = master_config.policy["train_global_batch_size"]
    min_trajectories_needed   = num_prompts_per_step

    print("📊 Buffer requirements calculation:")
    print(f"   - num_prompts_per_step: {num_prompts_per_step}")
    print(f"   - num_generations_per_prompt: {samples_per_prompt_group}")
    print(f"   - samples_per_prompt_group: {samples_per_prompt_group}")
    print(f"   - train_global_batch_size: {train_gbs}")
    print(f"   - min_trajectories_needed: {min_trajectories_needed} (async mode)")

    late_arrival_slack = 2
    optimal_buffer_size = (
        num_prompts_per_step * max_trajectory_age_steps * late_arrival_slack
    )

    # Both actors use sys.executable — no uv venvs in Dockyard.
    actor_runtime_env = {
        "py_executable": sys.executable,
        "env_vars":      dict(os.environ),
    }

    replay_buffer = ReplayBuffer.options(
        runtime_env=actor_runtime_env
    ).remote(max_size=optimal_buffer_size)

    # Ray actor handle: typed Any because Pylance's Ray stubs don't model the
    # per-method `.remote` accessor on this actor (the runtime provides it).
    trajectory_collector: Any = AsyncTrajectoryCollector.options(
        runtime_env=actor_runtime_env
    ).remote(
        policy_generation = policy_generation,
        tokenizer         = tokenizer,
        task_to_env       = task_to_env,
        master_config     = master_config,
        replay_buffer     = replay_buffer,
        start_step        = step,
    )

    collection_task = trajectory_collector.start_collection.remote(dataloader)
    trajectory_collector.set_weight_version.remote(weight_version)

    # Data-plane content store: register one persistent partition up front.
    # The ReplayBuffer remains the sampling/version coordinator (it holds the
    # metas), so balanced TQ sampling is bypassed (group_size=None) and the
    # trainer clears consumed sample ids per step via finish_step. No-op when
    # the data plane is disabled (colocated in-buffer path).
    _dp_cfg = getattr(master_config, "data_plane", None)
    _dp_enabled = bool(_dp_cfg) and bool(_dp_cfg.get("enabled", False))
    if _dp_enabled:
        policy.prepare_step(
            num_samples=optimal_buffer_size * samples_per_prompt_group,
            group_size=None,
        )

    print("📦 Started continuous background trajectory collection")
    print(
        f"🚀 Starting async GRPO training with "
        f"buffer_size={optimal_buffer_size}, max_age={max_trajectory_age_steps} steps"
    )

    print("⏳ Preparing policy generation for training...")
    if NEED_REFIT and weight_synchronizer.is_stale:
        print("🔄 Refitting policy generation with actual model weights...")
        try:
            weight_synchronizer.sync_weights()
            print("✅ Policy generation refit completed successfully")
        except Exception as e:
            print(f"❌ Policy generation refit failed: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("🔄 Preparing policy generation for inference...")
        try:
            policy_generation.prepare_for_generation()
            print("✅ Policy generation preparation completed successfully")
        except Exception as e:
            print(f"❌ Policy generation preparation failed: {e}")
            import traceback
            traceback.print_exc()
            return

    print("✅ Policy generation setup complete, proceeding to validation...")

    if val_at_start and step == 0:
        print("\n🔍 Running initial validation...")
        trajectory_collector.pause.remote()
        try:
            val_metrics, validation_timings = validate(
                policy_generation,
                val_dataloader,
                tokenizer,
                val_task_to_env,
                step=0,
                master_config=master_config,
                logger=logger,
            )
            policy_generation.finish_generation()
            logger.log_metrics(val_metrics, step, prefix="validation")
            logger.log_metrics(validation_timings, step, prefix="timing/validation")
            print("✅ Initial validation completed successfully")
        except Exception as e:
            print(f"❌ Initial validation failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            trajectory_collector.resume.remote()

    print("✅ All setup complete, starting buffer wait...")
    if policy_generation is not None:
        policy_generation.clear_logger_metrics()

    print(
        f"⏳ Waiting for replay buffer to have sufficient trajectories "
        f"({min_trajectories_needed} trajectories)..."
    )
    wait_iterations = 0
    while True:
        buffer_size_current = cast(int, ray.get(replay_buffer.size.remote()))  # type: ignore[union-attr]
        print(
            f"  Wait iteration {wait_iterations}: "
            f"buffer_filled_ratio={buffer_size_current}/{min_trajectories_needed}"
        )
        if buffer_size_current >= min_trajectories_needed:
            break
        time.sleep(1.0)
        wait_iterations += 1

    print("✅ Buffer ready! Starting training loop...")

    try:
        while step < master_config.grpo["max_num_steps"]:
            print(
                f"\n{'=' * 25} Step {step + 1}/"
                f"{master_config.grpo['max_num_steps']} {'=' * 25}"
            )
            maybe_gpu_profile_step(policy, step + 1)
            if policy != policy_generation and hasattr(policy_generation, "start_gpu_profiling"):
                maybe_gpu_profile_step(cast(Any, policy_generation), step + 1)

            with timer.time("total_step_time"):
                # Sample from replay buffer
                print("📦 Sampling from replay buffer...")
                with timer.time("exposed_generation"):
                    buffer_size_current = cast(int, ray.get(replay_buffer.size.remote()))  # type: ignore[union-attr]
                    print(
                        f"📊 Step coordination: training_step={step}, "
                        f"max_age={max_trajectory_age_steps}, "
                        f"buffer_size={buffer_size_current}"
                    )

                    num_prompt_groups_needed = master_config.grpo["num_prompts_per_step"]
                    sample_result: Optional[dict[str, Any]] = cast(
                        Optional[dict[str, Any]],
                        ray.get(
                            replay_buffer.sample.remote(  # type: ignore[union-attr]
                                num_prompt_groups      = num_prompt_groups_needed,
                                current_weight_version = weight_version,
                                max_age_steps          = max_trajectory_age_steps,
                            )
                        ),
                    )

                    if (
                        sample_result is None
                        or len(sample_result["trajectories"]) != num_prompt_groups_needed
                    ):
                        print(
                            "⏳ Buffer empty or not enough groups to form a full step, waiting..."
                        )
                        buffer_debug: dict[str, Any] = ray.get(replay_buffer.get_debug_info.remote())  # type: ignore[union-attr]
                        buffer_size  = buffer_debug["total_trajectories"]
                        if buffer_size > 0:
                            print(
                                f"🔍 Debug: Buffer has {buffer_size} trajectories but "
                                f"sampling requires exactly {num_prompt_groups_needed}."
                            )
                            print(f"   Current weight version: {weight_version}")
                            print(f"   Max trajectory age: {max_trajectory_age_steps}")
                            print(
                                f"   Trajectory versions in buffer: "
                                f"{buffer_debug['trajectory_versions']}"
                            )
                        time.sleep(0.5)
                        continue

                    trajectories       = sample_result["trajectories"]
                    avg_trajectory_age = cast(float, sample_result["avg_trajectory_age"])
                    print(
                        f"✅ Sampled {len(trajectories)} trajectory groups from buffer "
                        f"(avg age: {avg_trajectory_age:.2f} steps)"
                    )

                    per_group_metrics: dict[str, list] = {}
                    for t in trajectories:
                        for k, v in t["rollout_metrics"].items():
                            per_group_metrics.setdefault(k, []).append(v)
                    rollout_metrics = aggregate_rollout_metrics(per_group_metrics)

                    step_meta = None
                    driver_carry = None
                    repeated_batch = None
                    train_data = None
                    flat_messages = None
                    rewards = None
                    input_lengths = None
                    _dp_payload = None
                    if _dp_enabled:
                        # Bulk lives on the plane; concat per-group metas + the
                        # lightweight driver_carry. repeated_batch is unused.
                        _metas = [t["meta"] for t in trajectories]
                        step_meta = _metas[0].concat(*_metas[1:])
                        driver_carry = BatchedDataDict.from_batches(
                            [t["driver_carry"] for t in trajectories]
                        )
                        batch_n = len(step_meta.sample_ids)
                    else:
                        per_prompt_batches = [t["batch"] for t in trajectories]
                        repeated_batch = BatchedDataDict.from_batches(per_prompt_batches)
                        batch_n = repeated_batch.size

                expected_batch_size = (
                    master_config.grpo["num_prompts_per_step"]
                    * master_config.grpo["num_generations_per_prompt"]
                )
                if batch_n != expected_batch_size:
                    print(
                        f"❌ Unexpected training batch size: got {batch_n}, "
                        f"expected {expected_batch_size}. Skipping step."
                    )
                    time.sleep(0.5)
                    continue

                dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
                if expected_batch_size % dp_size != 0:
                    raise AssertionError(
                        f"Configuration error: (num_prompts_per_step * "
                        f"num_generations_per_prompt) = {expected_batch_size} must be "
                        f"divisible by data_parallel size {dp_size}."
                    )

                if _dp_enabled:
                    (
                        train_results,
                        max_seq_mult_prob_error,
                        num_masked_seqs,
                        masked_correct_pct,
                        _dp_payload,
                    ) = _dataplane_train_step(
                        policy,
                        step_meta,
                        driver_carry,
                        master_config=master_config,
                        adv_estimator=adv_estimator,
                        loss_fn=loss_fn,
                        weight_synchronizer=weight_synchronizer,
                        timer=timer,
                    )
                else:
                    _dp_payload = None
                    assert repeated_batch is not None
                    print(f"Got trajectory batch (size: {repeated_batch.size})")

                    # Rewards
                    print("▶ Processing rewards...")
                    with timer.time("reward_calculation"):
                        initial_prompt_message_logs = extract_initial_prompt_messages(
                            repeated_batch["message_log"],
                            repeated_batch["length"],
                        )
                        prompt_batched_flat, _ = batched_message_log_to_flat_message(
                            initial_prompt_message_logs,
                            pad_value_dict=cast(dict[str, int], {"token_ids": tokenizer.pad_token_id}),
                        )
                        prompt_ids_for_adv = prompt_batched_flat["token_ids"]
                        del initial_prompt_message_logs
                        del prompt_batched_flat

                        repeated_batch = apply_invalid_action_penalty(
                            repeated_batch,
                            master_config.grpo.get("invalid_action_penalty"),
                            step=step,
                        )
                        rewards = repeated_batch["total_reward"]
                        print(
                            f"  📊 Rewards stats: min={rewards.min():.4f}, "
                            f"max={rewards.max():.4f}, mean={rewards.mean():.4f}, "
                            f"std={rewards.std():.4f}"
                        )

                    # Training data
                    with timer.time("data_processing"):
                        add_grpo_token_loss_masks_and_generation_logprobs(
                            repeated_batch["message_log"]
                        )

                        flat_messages, input_lengths = batched_message_log_to_flat_message(
                            repeated_batch["message_log"],
                            pad_value_dict=cast(dict[str, int], {"token_ids": tokenizer.pad_token_id}),
                            make_sequence_length_divisible_by=master_config.policy[
                                "make_sequence_length_divisible_by"
                            ],
                        )

                        train_data = BatchedDataDict[ClippedPGLossDataDict](
                            {
                                "input_ids":           flat_messages["token_ids"],
                                "input_lengths":       input_lengths,
                                "generation_logprobs": flat_messages["generation_logprobs"],
                                "token_mask":          flat_messages["token_loss_mask"],
                                "sample_mask":         repeated_batch["loss_multiplier"],
                            }
                        )
                        # Thread per-message multimodal tensors (pixel_values,
                        # image_grid_thw, …) into train_data so the VLM policy forward
                        # receives them. The async path reuses this single train_data for
                        # get_logprobs and policy.train, so one update covers both (the
                        # sync path does the equivalent across its two data dicts). On the
                        # text-only path get_multimodal_dict returns {} (inert).
                        extra_multimodal_data = flat_messages.get_multimodal_dict(
                            as_tensors=False
                        )
                        train_data.update(extra_multimodal_data)
                        train_data.to("cpu")

                    # Logprobs
                    skip_prev_logprobs = master_config.loss_fn.force_on_policy_ratio
                    fprop_logprobs     = torch.zeros_like(train_data["generation_logprobs"])
                    if skip_prev_logprobs:
                        print(
                            "▶ Skipping prev_logprobs (force_on_policy_ratio=True)...",
                            flush=True,
                        )
                    else:
                        print("▶ Preparing for logprob inference...")
                        with timer.time("logprob_inference_prep"):
                            policy.prepare_for_lp_inference()

                    print("▶ Computing logprobs...", flush=True)
                    with timer.time("policy_and_reference_logprobs"):
                        if not skip_prev_logprobs:
                            fprop_logprobs = policy.get_logprobs(
                                train_data, timer=timer
                            )["logprobs"]
                        train_data["prev_logprobs"] = fprop_logprobs

                        if not master_config.grpo.get(
                            "skip_reference_policy_logprobs_calculation"
                        ):
                            reference_logprobs = policy.get_reference_policy_logprobs(
                                train_data, timer=timer
                            )["reference_logprobs"]
                            train_data["reference_policy_logprobs"] = reference_logprobs
                        else:
                            train_data["reference_policy_logprobs"] = torch.zeros_like(
                                fprop_logprobs
                            )

                    seq_logprob_error_threshold = master_config.grpo.get(
                        "seq_logprob_error_threshold", None
                    )
                    if skip_prev_logprobs:
                        assert seq_logprob_error_threshold is None, (
                            "seq_logprob_error_threshold requires prev_logprobs; "
                            "cannot use with force_on_policy_ratio=True"
                        )
                        max_seq_mult_prob_error = 0.0
                        num_masked_seqs        = 0
                        masked_correct_pct     = 0.0
                    else:
                        (
                            max_seq_mult_prob_error,
                            num_masked_seqs,
                            masked_correct_pct,
                        ) = compute_and_apply_seq_logprob_error_masking(
                            train_data=train_data,
                            rewards=rewards,
                            seq_logprob_error_threshold=seq_logprob_error_threshold,
                        )

                    # Advantages
                    with timer.time("advantage_calculation"):
                        print("▶ Computing advantages...", flush=True)
                        token_mask  = train_data["token_mask"]
                        sample_mask = train_data["sample_mask"]
                        mask        = token_mask * sample_mask.unsqueeze(-1)

                        train_data["advantages"] = adv_estimator.compute_advantage(
                            prompt_ids         = prompt_ids_for_adv,
                            rewards            = rewards,
                            mask               = mask,
                            repeated_batch     = repeated_batch,
                            logprobs_policy    = train_data["prev_logprobs"],
                            logprobs_reference = train_data.get(
                                "reference_policy_logprobs"
                            ),
                        )
                        # Advantage-locus invalid-action penalties on flagged spans (N2).
                        (
                            train_data["advantages"],
                            _,
                        ) = apply_message_span_advantage_penalties(
                            train_data["advantages"],
                            repeated_batch["message_log"],
                            master_config.grpo.get("invalid_action_penalty"),
                            step=step,
                        )
                        train_data["advantages"] = _clip_grpo_advantages(
                            train_data["advantages"], master_config.grpo
                        )
                        del prompt_ids_for_adv

                        advantages = train_data["advantages"]
                        print(
                            f"  📊 Advantages stats: min={advantages.min():.4f}, "
                            f"max={advantages.max():.4f}, mean={advantages.mean():.4f}, "
                            f"std={advantages.std():.4f}"
                        )

                    # Training step
                    print("▶ Preparing for training...")
                    with timer.time("training_prep"):
                        policy.prepare_for_training()
                        weight_synchronizer.mark_stale()

                    print("▶ Training policy...")
                    with timer.time("policy_training"):
                        train_results = policy.train(train_data, loss_fn, timer=timer)

                # Weight sync
                print("🔄 Synchronizing policy weights to trajectory collector…")
                generation_logger_metrics = None
                if NEED_REFIT:
                    print("🔄 Coordinating with trajectory collector before refit...")
                    with timer.time("exposed_generation"):
                        ray.get(trajectory_collector.prepare_for_refit.remote())

                    if policy_generation is not None:
                        generation_logger_metrics = (
                            policy_generation.get_logger_metrics()
                        )

                    print("🔄 Performing policy generation refit...")
                    with timer.time("weight_sync"):
                        weight_synchronizer.sync_weights()
                        weight_version += 1
                        trajectory_collector.set_weight_version.remote(weight_version)
                        trajectory_collector.resume_after_refit.remote()

                if policy_generation is not None:
                    policy_generation.clear_logger_metrics()

                # Validation
                val_metrics, validation_timings = None, None
                is_last_step = step + 1 == master_config.grpo["max_num_steps"]

                if (val_period > 0 and (step + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
                    trajectory_collector.pause.remote()

                    if NEED_REFIT and weight_synchronizer.is_stale:
                        weight_synchronizer.sync_weights()
                    else:
                        policy_generation.prepare_for_generation()

                    val_metrics, validation_timings = validate(
                        policy_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=step + 1,
                        master_config=master_config,
                        logger=logger,
                    )
                    policy_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, step + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(val_metrics, step + 1, prefix="validation")

                    gc.collect()
                    torch.cuda.empty_cache()
                    trajectory_collector.resume.remote()

                # Metrics
                if _dp_enabled:
                    # Bulk stayed on the plane; source per-token/per-row values
                    # from the driver's read-back (no input_ids on the driver).
                    assert _dp_payload is not None
                    flat_advantages       = _dp_payload["advantages"]
                    flat_token_mask       = _dp_payload["token_mask"]
                    flat_messages_content = []
                    rewards               = _dp_payload["rewards"]
                    input_lengths         = _dp_payload["input_lengths"]
                    mean_prompt_length    = _dp_payload["length"]
                else:
                    assert train_data is not None
                    assert flat_messages is not None
                    assert repeated_batch is not None
                    assert rewards is not None
                    assert input_lengths is not None
                    flat_advantages       = train_data["advantages"]
                    flat_token_mask       = flat_messages["token_loss_mask"]
                    flat_messages_content = flat_messages.get("content", [])
                    del flat_messages
                    mean_prompt_length    = repeated_batch["length"]

                response_advantages = torch.masked_select(
                    flat_advantages, flat_token_mask.bool()
                )

                metrics: dict = {
                    "loss":               train_results["loss"].numpy(),
                    "reward":             rewards.numpy(),
                    "grad_norm":          train_results["grad_norm"].numpy(),
                    "mean_prompt_length": mean_prompt_length.numpy(),
                    "total_num_tokens":   input_lengths.numpy(),
                    "advantages/mean":    torch.mean(response_advantages).detach().item()
                    if response_advantages.numel() > 0 else 0.0,
                    "advantages/max":     torch.max(response_advantages).detach().item()
                    if response_advantages.numel() > 0 else 0.0,
                    "advantages/min":     torch.min(response_advantages).detach().item()
                    if response_advantages.numel() > 0 else 0.0,
                }
                if "moe_metrics" in train_results:
                    metrics.update(
                        {f"moe/{k}": v for k, v in train_results["moe_metrics"].items()}
                    )
                metrics.update(train_results["all_mb_metrics"])

                for k, v in metrics.items():
                    if k in {"probs_ratio_min", "probs_ratio_clamped_min"}:
                        valid_values = [x for x in v if not np.isinf(x)]
                        metrics[k]   = (
                            np.min(valid_values).item() if valid_values else -1.0
                        )
                    elif k in {"probs_ratio_max", "probs_ratio_clamped_max"}:
                        valid_values = [x for x in v if not np.isinf(x)]
                        metrics[k]   = (
                            np.max(valid_values).item() if valid_values else -1.0
                        )
                    elif k in {
                        "lr", "wd", "reward", "global_valid_seqs",
                        "global_valid_toks", "mean_prompt_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()

                metrics.update(rollout_metrics)
                if generation_logger_metrics is not None:
                    metrics["generation_logger_metrics"] = generation_logger_metrics
                total_valid_tokens                          += metrics["global_valid_toks"]
                metrics["max_seq_mult_prob_error"]           = max_seq_mult_prob_error
                metrics["num_masked_seqs_by_logprob_error"] = num_masked_seqs
                metrics["masked_correct_pct"]                = masked_correct_pct

                # Checkpointing
                consumed_samples += master_config.grpo["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (step + 1) % master_config.checkpointing["save_period"] == 0
                )
                should_save_by_timeout = timeout.check_save()

                if master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    grpo_save_state["current_step"]       = step + 1
                    grpo_save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        grpo_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in grpo_save_state:
                        del grpo_save_state["val_reward"]
                    grpo_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config.checkpointing["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with "
                            "'val:' or 'train:'."
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on "
                                f"{metric_name} but no {prefix} metrics were "
                                "collected. This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in grpo_save_state:
                                del grpo_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            grpo_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {step + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            step + 1, grpo_save_state, master_config
                        )
                        policy.save_checkpoint(
                            weights_path   = os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path = (
                                os.path.join(
                                    checkpoint_path, "policy", "optimizer"
                                )
                                if checkpointer.save_optimizer
                                else None
                            ),
                            tokenizer_path = os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg = master_config.checkpointing,
                        )
                        actual_dataloader_state = ray.get(
                            trajectory_collector.get_dataloader_state.remote()
                        )
                        torch.save(
                            actual_dataloader_state,
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # Logging
            log_data: dict = {}
            if _dp_enabled:
                # input_ids stays on the plane (workers fetch it for train), so
                # the per-step jsonl omits the token-id bulk; per-token deltas
                # and masks come from the driver read-back.
                assert _dp_payload is not None
                log_data["content"]             = flat_messages_content
                log_data["rewards"]             = _dp_payload["rewards"].tolist()
                log_data["input_lengths"]       = _dp_payload["input_lengths"].tolist()
                log_data["token_loss_mask"]     = _dp_payload["token_mask"].tolist()
                log_data["sample_loss_mask"]    = _dp_payload["sample_mask"].tolist()
                log_data["advantages"]          = _dp_payload["advantages"].tolist()
                log_data["generation_logprobs"] = _dp_payload["generation_logprobs"].tolist()
                log_data["prev_logprobs"]       = _dp_payload["prev_logprobs"].tolist()
            else:
                assert repeated_batch is not None
                assert rewards is not None
                assert input_lengths is not None
                assert train_data is not None
                if "agent_ref" in repeated_batch:
                    log_data["agent_ref"] = repeated_batch["agent_ref"]
                log_data["content"]          = flat_messages_content
                log_data["rewards"]          = rewards.tolist()
                if master_config.grpo["use_dynamic_sampling"]:
                    log_data["filtered_rewards"] = rewards.tolist()
                    log_data["rewards"]          = repeated_batch["total_reward"].tolist()
                log_data["input_lengths"]      = input_lengths.tolist()
                log_data["token_ids"]          = train_data["input_ids"].tolist()
                log_data["token_loss_mask"]    = train_data["token_mask"].tolist()
                log_data["sample_loss_mask"]   = train_data["sample_mask"].tolist()
                log_data["advantages"]         = train_data["advantages"].tolist()
                log_data["generation_logprobs"] = train_data["generation_logprobs"].tolist()
                log_data["prev_logprobs"]       = train_data["prev_logprobs"].tolist()
                del train_data
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{step + 1}.jsonl"
            )
            del flat_messages_content

            if _dp_enabled:
                # Drop this step's consumed samples from the plane's persistent
                # partition (the ReplayBuffer already popped the metas). Without
                # this the rolling partition grows unbounded.
                policy.finish_step(step_meta)

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )

            buffer_size_current = cast(int, ray.get(replay_buffer.size.remote()))  # type: ignore[union-attr]
            metrics["buffer_size"]          = buffer_size_current
            metrics["avg_trajectory_age"]   = avg_trajectory_age

            if (
                master_config.policy["generation"]
                .get("vllm_cfg", {})
                .get("enable_vllm_metrics_logger", False)
                and master_config.logger["wandb_enabled"]
                and generation_logger_metrics is not None
            ):
                log_generation_metrics_to_wandb(
                    generation_logger_metrics,
                    step + 1,
                    master_config.policy["generation"]["vllm_cfg"][
                        "vllm_metrics_logger_interval"
                    ],
                    logger,
                )

            if (
                master_config.policy["generation"]
                .get("vllm_cfg", {})
                .get("async_engine", False)
            ):
                for metric_name in metrics.keys():
                    if metric_name.startswith("histogram/"):
                        logger.log_histogram(
                            metrics[metric_name],
                            step + 1,
                            f"generation_metrics/{metric_name}",
                        )

            print("\n📊 Training Results:")
            print(f"  • Loss: {metrics['loss']:.4f}")
            if "draft_loss" in metrics:
                print(f"  • Draft Loss: {metrics['draft_loss']:.4f}")
            print(f"  • Generation KL Error: {metrics['gen_kl_error']:.4f}")
            assert rewards is not None
            print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
            print(f"  • Buffer Size: {buffer_size_current}")
            print(f"  • Avg Trajectory Age: {avg_trajectory_age:.2f} steps")

            print("\n⏱️  Timing:")
            total_time = timing_metrics.get("total_step_time", 0)
            print(f"  • Total step time: {total_time:.2f}s")
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            total_num_gpus = (
                master_config.cluster["num_nodes"]
                * master_config.cluster["gpus_per_node"]
            )
            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            performance_metrics = print_performance_metrics(
                train_results, metrics, timing_metrics, master_config.model_dump()
            )

            logger.log_metrics(performance_metrics, step + 1, prefix="performance")
            logger.log_metrics(metrics, step + 1, prefix="train")
            logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

            timer.reset()
            step += 1
            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if step >= master_config.grpo["max_num_steps"]:
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

    except Exception as e:
        print(f"❌ Error in async loop: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("🛑 Stopping trajectory collection...")
        try:
            ray.kill(trajectory_collector)
        except Exception as e:
            print(f"Error stopping trajectory collector: {e}")
        try:
            ray.kill(replay_buffer)  # type: ignore[arg-type]
        except Exception as e:
            print(f"Error stopping replay buffer: {e}")
        print("Async GRPO training complete!")
