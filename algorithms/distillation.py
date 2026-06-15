import os
import warnings
from typing import TYPE_CHECKING, Any, NotRequired, Optional, TypedDict, cast
import numpy as np
import torch
from pydantic import BaseModel
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer
from dockyard_rl.algorithms.loss.loss_functions import DistillationLossFn
from dockyard_rl.algorithms.utils import set_seed
from dockyard_rl.data import DataConfig
from dockyard_rl.data.collate_fn import rl_collate_fn
from dockyard_rl.data.datasets import AllTaskProcessedDataset
from dockyard_rl.data.llm_message_utils import add_loss_mask_to_message_log
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from dockyard_rl.models.policy import PolicyConfig
from dockyard_rl.models.policy.lm_policy import Policy

if TYPE_CHECKING:
    from dockyard_rl.utils.checkpoint import CheckpointingConfig
    from dockyard_rl.utils.logger import LoggerConfig

from dockyard_rl.utils.checkpoint import CheckpointManager
from dockyard_rl.utils.logger import Logger
from dockyard_rl.utils.memory_tracker import MemoryTracker

try:
    from dockyard_rl.utils.nsys import maybe_gpu_profile_step
except ImportError:
    def maybe_gpu_profile_step(*args, **kwargs) -> None:  # type: ignore[misc]
        pass

try:
    from dockyard_rl.utils.timer import TimeoutChecker, Timer
except ImportError:
    raise ImportError(
        "dockyard_rl.utils.timer is required for distillation.py but has not been ported yet."
    )

# Rollout infrastructure — deferred until experience/ is ported
try:
    from dockyard_rl.experience.rollouts import (  # type: ignore[import]
        run_async_multi_turn_rollout,
        run_multi_turn_rollout,
    )
except ImportError:
    run_multi_turn_rollout = None  # type: ignore[assignment]
    run_async_multi_turn_rollout = None  # type: ignore[assignment]

class DistillationSaveState(TypedDict):
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    val_loss: NotRequired[float]
    consumed_samples: int
    total_valid_tokens: int  # Track total number of non-padding tokens during training

def _default_distillation_save_state() -> DistillationSaveState:
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }

class DistillationConfig(TypedDict):
    max_num_steps: int
    max_num_epochs: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    val_at_end: bool
    seed: int
    # Number of top-k logits to transfer from teacher to student.
    # Higher values increase memory usage but improve distillation fidelity.
    # Recommended range: 512–4096.
    num_topk_logits: int

class TeacherConfig(TypedDict):
    """Config for the frozen teacher policy."""

    model_name: str
    resources: dict[str, Any]
    # All other Policy config fields are inherited from the teacher policy_config
    # passed to setup(). Only resource allocation and model name differ.
    dtype: NotRequired[str]


class MasterConfig(BaseModel, extra="allow"):
    policy: PolicyConfig          # Student policy config
    teacher_policy: PolicyConfig  # Teacher policy config (frozen)
    data: DataConfig
    distillation: DistillationConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig

# ==============================================================================
# Setup & Initialization
# ==============================================================================
def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
    task_to_env: dict[str, Any],
    val_task_to_env: dict[str, Any],
) -> tuple[
    Policy,
    Policy,
    RayVirtualCluster,
    RayVirtualCluster,
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    DistillationLossFn,
    Logger,  # type: ignore[valid-type]
    CheckpointManager,  # type: ignore[valid-type]
    Optional[DistillationSaveState],
    MasterConfig,
]:
    """Main entry point for running on-policy distillation.

    The student generates rollouts against environments. The teacher scores
    those rollouts by providing top-k logits. The student is trained to match
    the teacher distribution via DistillationLossFn.

    Returns:
        Tuple of student_policy, teacher_policy, student_cluster, teacher_cluster,
        train_dataloader, val_dataloader, loss_fn, logger, checkpointer,
        distillation_save_state, master_config
    """
    set_seed(master_config.distillation["seed"])

    student_policy_config = master_config.policy
    teacher_policy_config = master_config.teacher_policy
    data_config = master_config.data
    distillation_config = master_config.distillation
    logger_config = master_config.logger
    cluster_config = master_config.cluster
    checkpointing_config = master_config.checkpointing


    # ==================================
    # Logger
    # ==================================
    logger = Logger(logger_config)  # type: ignore[call-arg]
    logger.log_hyperparams(master_config.model_dump())

    # ==================================
    # Checkpointing
    # ==================================
    checkpointer = CheckpointManager(checkpointing_config)  # type: ignore[call-arg]
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    distillation_save_state: Optional[DistillationSaveState] = cast(
        Optional[DistillationSaveState],
        checkpointer.load_training_info(last_checkpoint_path),
    )

    # ==================================
    # Data
    # ==================================
    train_gbs = student_policy_config.get("train_global_batch_size")
    assert train_gbs is not None, "PolicyConfig.train_global_batch_size is required for training"
    train_dataloader = StatefulDataLoader(
        train_dataset,  # type: ignore[arg-type]
        batch_size=train_gbs,
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=data_config.get("num_workers", 0),
    )

    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        train_dataloader.load_state_dict(dataloader_state_dict)

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = StatefulDataLoader(
            val_dataset,  # type: ignore[arg-type]
            batch_size=distillation_config["val_global_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
            drop_last=False,
            num_workers=data_config.get("num_workers", 0),
        )

    # ==================================
    # Clusters — student and teacher are on separate virtual clusters
    # so they can be placed on different node groups.
    # ==================================
    print("\n▶ Setting up compute clusters...")

    student_cluster = RayVirtualCluster(
        name="distillation_student_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )

    teacher_cluster_config = (master_config.model_extra or {}).get(
        "teacher_cluster", cluster_config
    )
    teacher_cluster = RayVirtualCluster(
        name="distillation_teacher_cluster",
        bundle_ct_per_node_list=[teacher_cluster_config["gpus_per_node"]]
        * teacher_cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=teacher_cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    print(
        f"  ✓ Student cluster: {cluster_config['num_nodes']} nodes × "
        f"{cluster_config['gpus_per_node']} GPUs"
    )
    print(
        f"  ✓ Teacher cluster: {teacher_cluster_config['num_nodes']} nodes × "
        f"{teacher_cluster_config['gpus_per_node']} GPUs"
    )

    # ==================================
    # Policies
    # ==================================
    print("\n▶ Setting up student policy...")
    weights_path, optimizer_path = checkpointer.get_resume_paths(last_checkpoint_path)
    if last_checkpoint_path is None:
        pretrained_cfg = checkpointing_config.get("pretrained_checkpoint")
        if pretrained_cfg is not None:
            weights_path, optimizer_path = checkpointer.get_pretrained_paths(pretrained_cfg)

    student_policy = Policy(
        cluster=student_cluster,
        config=student_policy_config,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,
    )
    student_policy.print_node_ip_and_gpu_id()
    print("  ✓ Student policy initialized")

    print("\n▶ Setting up teacher policy (frozen)...")
    teacher_policy = Policy(
        cluster=teacher_cluster,
        config=teacher_policy_config,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        weights_path=teacher_policy_config.get("weights_path", None),
        optimizer_path=None,
        init_optimizer=False,
        init_reference_model=False,
    )
    teacher_policy.print_node_ip_and_gpu_id()
    print("  ✓ Teacher policy initialized (no optimizer — frozen)")

    loss_fn = DistillationLossFn(num_topk_logits=distillation_config["num_topk_logits"])  # type: ignore[call-arg]

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        student_policy,
        teacher_policy,
        student_cluster,
        teacher_cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        distillation_save_state,
        master_config,
    )

# ==============================================================================
# Training & Validation
# ==============================================================================
def validate(
    student_policy: Policy,
    teacher_policy: Policy,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    loss_fn: DistillationLossFn,
    step: int,
    master_config: MasterConfig,
    val_batches: int,
    val_batch_size: int,
    val_mbs: int,
    logger: Logger,  # type: ignore[valid-type]
) -> tuple[dict, dict]:
    """Run validation — student generates, teacher scores, loss is computed."""
    if val_dataloader is None:
        assert master_config.distillation["val_period"] <= 0, (
            "val_dataloader is None, so distillation.val_period must be <= 0"
        )
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return {}, {}

    timer = Timer()

    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        val_metrics = {"val_loss": 0.0}
        sum_num_valid_tokens = 0

        student_policy.prepare_for_training()
        for batch_idx, val_batch in enumerate(val_dataloader):
            add_loss_mask_to_message_log(
                val_batch["message_log"],
                roles_to_train_on=["assistant"],
                only_unmask_final=True,
            )

            # Get teacher top-k logits for the student's context
            teacher_topk = teacher_policy.get_topk_logits(
                val_batch,
                micro_batch_size=val_mbs,
                num_topk_logits=master_config.distillation["num_topk_logits"],  # type: ignore[call-arg]
            )
            val_batch["teacher_topk_logits"] = teacher_topk["topk_logits"]
            val_batch["teacher_topk_indices"] = teacher_topk["topk_indices"]

            val_data = BatchedDataDict(val_batch)

            if val_data.size < val_batch_size:
                dp_size = student_policy.sharding_annotations.get_axis_size(
                    "data_parallel"
                )
                from dockyard_rl.algorithms.utils import maybe_pad_last_batch
                val_data = maybe_pad_last_batch(val_data, dp_size, val_mbs)

            val_results = student_policy.train(
                val_data,
                loss_fn,
                eval_mode=True,
                gbs=val_data.size,
                mbs=val_mbs,
            )

            if len(val_results["all_mb_metrics"]) == 0:
                warnings.warn(
                    "No validation metrics were collected for this batch."
                    " This is likely because there were no valid samples."
                )
            else:
                num_valid_tokens = (
                    val_data["sample_mask"].unsqueeze(-1) * val_data["token_mask"]
                ).sum()
                val_metrics["val_loss"] += (
                    float(val_results["loss"]) * num_valid_tokens
                )
                sum_num_valid_tokens += num_valid_tokens

            if val_batches > 0 and batch_idx >= val_batches - 1:
                break

        if sum_num_valid_tokens > 0:
            val_metrics["val_loss"] /= sum_num_valid_tokens
        else:
            warnings.warn(
                "No validation metrics were collected."
                " This is likely because there were no valid samples in the validation set."
            )

        student_policy.prepare_for_training()

    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    if sum_num_valid_tokens > 0:
        print("\n📊 Validation Results:")
        print(f"    • Validation loss: {val_metrics['val_loss']:.4f}")
        print("\n  ⏱️  Validation Timing:")
        print(f"    • Total validation time: {validation_time:.2f}s")

    logger.log_metrics(val_metrics, step, prefix="validation")
    logger.log_metrics(timing_metrics, step, prefix="timing/validation")

    timer.reset()
    return val_metrics, timing_metrics

def distillation_train(
    student_policy: Policy,
    teacher_policy: Policy,
    train_dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    loss_fn: DistillationLossFn,
    master_config: MasterConfig,
    logger: Logger,  # type: ignore[valid-type]
    checkpointer,
    distillation_save_state: Optional[DistillationSaveState],
    task_to_env: dict[str, Any],
    val_task_to_env: dict[str, Any],
) -> None:
    """Main distillation training loop.

    Each step:
    1. Sample a prompt batch from train_dataloader.
    2. Run on-policy rollouts with the student via run_multi_turn_rollout.
    3. Query teacher for top-k logits on the student's generated sequences.
    4. Train student to minimise DistillationLossFn against teacher logits.

    Args:
        student_policy:           The trainable student policy.
        teacher_policy:           The frozen teacher policy (no optimizer).
        train_dataloader:         Prompt dataloader.
        val_dataloader:           Optional validation prompt dataloader.
        tokenizer:                Shared tokenizer (same for both policies).
        loss_fn:                  DistillationLossFn instance.
        master_config:            Full experiment config.
        logger:                   Metrics logger.
        checkpointer:             CheckpointManager instance.
        distillation_save_state:  Resumable training state dict.
        task_to_env:              Mapping task_name → environment actor for training.
        val_task_to_env:          Mapping task_name → environment actor for validation.
    """
    assert run_multi_turn_rollout is not None, (
        "dockyard_rl.experience.rollouts must be ported before distillation_train() "
        "can be used. run_multi_turn_rollout is None."
    )

    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config.checkpointing.get("checkpoint_must_save_by"),
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    if distillation_save_state is None:
        distillation_save_state = _default_distillation_save_state()
        current_epoch = 0
        current_step = 0
        total_steps = 0
        total_valid_tokens = 0
    else:
        current_epoch = distillation_save_state["epoch"]
        current_step = distillation_save_state["step"]
        total_steps = distillation_save_state["total_steps"]
        total_valid_tokens = distillation_save_state.get("total_valid_tokens", 0)

    distillation_config = master_config.distillation
    val_period = distillation_config["val_period"]
    val_at_start = distillation_config["val_at_start"]
    val_at_end = distillation_config["val_at_end"]
    max_num_epochs = distillation_config["max_num_epochs"]
    num_topk_logits = distillation_config["num_topk_logits"]

    if val_at_start and total_steps == 0:
        print("\n📍 Running initial validation...")
        validate(
            student_policy,
            teacher_policy,
            val_dataloader,
            tokenizer,
            loss_fn,
            step=0,
            master_config=master_config,
            val_batches=distillation_config["val_batches"],
            val_batch_size=distillation_config["val_global_batch_size"],
            val_mbs=distillation_config["val_micro_batch_size"],
            logger=logger,
        )

    student_policy.prepare_for_training()

    # Determine whether to use the async rollout path (configured via grpo-style flags
    # in the policy config, reused here since we share the same infrastructure).
    use_async_rollouts = master_config.policy.get("use_async_rollouts", False)

    while (
        current_epoch < max_num_epochs
        and total_steps < master_config.distillation["max_num_steps"]
    ):
        print(
            f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}"
        )

        for batch in train_dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/"
                f"{min(len(train_dataloader), master_config.distillation['max_num_steps'])} "
                f"{'=' * 25}"
            )
            maybe_gpu_profile_step(student_policy, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # ----------------------------------------
                # On-policy rollouts (student generates)
                # ----------------------------------------
                print("▶ Running student rollouts...")
                with timer.time("rollout"):
                    from typing import Callable
                    _async_rollout_fn = cast(Callable[..., Any], run_async_multi_turn_rollout)
                    _rollout_fn       = cast(Callable[..., Any], run_multi_turn_rollout)
                    if use_async_rollouts:
                        rollout_batch, rollout_metrics = _async_rollout_fn(
                            policy=student_policy,
                            data=batch,
                            task_to_env=task_to_env,
                            tokenizer=tokenizer,
                            master_config=master_config,
                            timer=timer,
                        )
                    else:
                        rollout_batch, rollout_metrics = _rollout_fn(
                            policy=student_policy,
                            data=batch,
                            task_to_env=task_to_env,
                            tokenizer=tokenizer,
                            master_config=master_config,
                            timer=timer,
                        )

                # ----------------------------------------
                # Loss masking — only unmask final assistant turn
                # (same convention as grpo.py for RL rollouts)
                # ----------------------------------------
                add_loss_mask_to_message_log(
                    rollout_batch["message_log"],
                    roles_to_train_on=["assistant"],
                    only_unmask_final=True,
                )

                # ----------------------------------------
                # Teacher top-k logits
                # ----------------------------------------
                print("▶ Querying teacher for top-k logits...")
                with timer.time("teacher_logits"):
                    train_mbs = master_config.policy.get("train_micro_batch_size")
                    assert train_mbs is not None, "PolicyConfig.train_micro_batch_size is required for training"
                    teacher_topk = teacher_policy.get_topk_logits(
                        rollout_batch,
                        micro_batch_size=train_mbs,
                        num_topk_logits=num_topk_logits,  # type: ignore[call-arg]
                    )
                    rollout_batch["teacher_topk_logits"] = teacher_topk["topk_logits"]
                    rollout_batch["teacher_topk_indices"] = teacher_topk["topk_indices"]

                # ----------------------------------------
                # Student training step
                # ----------------------------------------
                print("▶ Taking student training step...")
                with timer.time("policy_training"):
                    train_results = student_policy.train(
                        BatchedDataDict(rollout_batch),
                        loss_fn,
                        timer=timer,
                    )

                is_last_step = total_steps + 1 >= master_config.distillation[
                    "max_num_steps"
                ] or (
                    current_epoch + 1 == max_num_epochs
                    and current_step + 1 == len(train_dataloader)
                )

                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
                    val_metrics, validation_timings = validate(
                        student_policy,
                        teacher_policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        step=total_steps + 1,
                        master_config=master_config,
                        val_batches=distillation_config["val_batches"],
                        val_batch_size=distillation_config["val_global_batch_size"],
                        val_mbs=distillation_config["val_micro_batch_size"],
                        logger=logger,
                    )

                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                }
                if "moe_metrics" in train_results:
                    metrics.update(
                        {
                            f"moe/{k}": v
                            for k, v in train_results["moe_metrics"].items()
                        }
                    )
                metrics.update(train_results["all_mb_metrics"])
                metrics.update(rollout_metrics)
                for k, v in metrics.items():
                    if k in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
                        metrics[k] = np.mean(v).item()
                    else:
                        try:
                            metrics[k] = np.sum(v).item()
                        except (TypeError, AttributeError):
                            pass
                total_valid_tokens += metrics.get("global_valid_toks", 0)

                policy_train_gbs = master_config.policy.get("train_global_batch_size")
                assert policy_train_gbs is not None, "PolicyConfig.train_global_batch_size is required for training"
                distillation_save_state["consumed_samples"] += policy_train_gbs
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1)
                    % master_config.checkpointing["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()

                if master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    distillation_save_state["step"] = (current_step + 1) % len(
                        train_dataloader
                    )
                    distillation_save_state["total_steps"] = total_steps + 1
                    distillation_save_state["epoch"] = current_epoch
                    distillation_save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        distillation_save_state.update(val_metrics)

                    full_metric_name = master_config.checkpointing["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with "
                            f"'val:' or 'train:'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = (
                            metrics if prefix == "train" else val_metrics
                        )
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on "
                                f"{metric_name} but no {prefix} metrics were "
                                f"collected. This checkpoint will not be saved "
                                f"as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in distillation_save_state:
                                del distillation_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            distillation_save_state[full_metric_name] = (
                                metrics_source[metric_name]
                            )

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {total_steps + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1,
                            distillation_save_state,
                            master_config,
                        )
                        # Save only the student — teacher is frozen and loaded
                        # from its original weights_path on resume.
                        student_policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            )
                            if checkpointer.save_optimizer
                            else None,
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config.checkpointing,
                        )
                        torch.save(
                            train_dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            timing_metrics = timer.get_timing_metrics(reduction_op="sum")

            print("\n📊 Training Results:")
            print(f"  • Loss: {float(metrics['loss']):.4f}")
            print(f"  • Grad norm: {float(metrics['grad_norm']):.4f}")

            print("\n⏱️  Timing:")
            total_time_raw = timing_metrics.get("total_step_time", 0)
            total_time = total_time_raw if isinstance(total_time_raw, float) else 0.0
            print(f"  • Total step time: {total_time:.2f}s")
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    v_scalar = v if isinstance(v, float) else 0.0
                    percent = (v_scalar / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v_scalar:.2f}s ({percent:.1f}%)")

            total_num_gpus = (
                master_config.cluster["num_nodes"]
                * master_config.cluster["gpus_per_node"]
            )
            if total_time > 0:
                timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                    metrics.get("global_valid_toks", 0)
                    / total_time
                    / total_num_gpus
                )
            else:
                timing_metrics["valid_tokens_per_sec_per_gpu"] = 0.0

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(
                timing_metrics, total_steps + 1, prefix="timing/train"
            )

            timer.reset()
            current_step += 1
            total_steps += 1

            if should_save_by_timeout:
                print(
                    "Timeout has been reached, stopping training early", flush=True
                )
                return
            if total_steps >= master_config.distillation["max_num_steps"]:
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        current_epoch += 1
        current_step = 0