import os
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Optional, Union
import numpy as np
import ray
import torch
from ray.util.queue import Queue as RayQueue
from transformers import AutoProcessor, PreTrainedTokenizerBase
from dockyard_rl.algorithms.loss.interfaces import LossFunction
from dockyard_rl.distributed.batched_data_dict import (
    BatchedDataDict,
    DynamicBatchingArgs,
    SequencePackingArgs,
    SlicedDataDict,
)
from dockyard_rl.distributed.named_sharding import NamedSharding
from dockyard_rl.distributed.virtual_cluster import RayVirtualCluster
from dockyard_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from dockyard_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from dockyard_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
    ReferenceLogprobOutputSpec,
    ScoreOutputSpec,
    TopkLogitsOutputSpec,
)

# JAX trainer backend selection. config.py is pure-stdlib (no jax/flax import),
# so this is safe to import in the torch-only driver process.
from dockyard_rl.models.jax.config import jax_backend_enabled

# Deferred: written as part of the policy workers layer
try:
    from dockyard_rl.models.policy import PolicyConfig
except ImportError:
    PolicyConfig = dict  # type: ignore

try:
    from dockyard_rl.models.policy.utils import resolve_policy_worker_cls
except ImportError:
    def resolve_policy_worker_cls(fqn: str, config: Any) -> str:  # type: ignore
        return fqn

# Deferred: utils/checkpoint.py (#14 in backlog)
try:
    from dockyard_rl.utils.checkpoint import CheckpointingConfig
except ImportError:
    CheckpointingConfig = dict  # type: ignore

# Deferred: utils/timer.py (#18 in backlog)
try:
    from dockyard_rl.utils.timer import Timer
except ImportError:
    Timer = None  # type: ignore

PathLike = Union[str, "os.PathLike[Any]"]

class Policy(ColocatablePolicyInterface, GenerationInterface):
    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: PolicyConfig,  # type: ignore[valid-type]
        tokenizer: PreTrainedTokenizerBase,
        name_prefix: str = "lm_policy",
        workers_per_node: Optional[Union[int, list[int]]] = None,
        init_optimizer: bool = True,
        weights_path: Optional[PathLike] = None,
        optimizer_path: Optional[PathLike] = None,
        init_reference_model: bool = True,
        processor: Optional[AutoProcessor] = None,
        worker_extension_cls_fqn: Optional[str] = None,
    ):
        if weights_path:
            weights_path = os.path.abspath(weights_path)
        if optimizer_path:
            optimizer_path = os.path.abspath(optimizer_path)

        worker_builder_cls_fqn: str
        tp_size = 1
        pp_size = 1
        cp_size = 1
        use_v2 = False

        dtensor_enable = bool(config.get("dtensor_cfg", {}).get("enabled", False))
        jax_enable = jax_backend_enabled(config)
        draft_enabled = bool(config.get("draft", {}).get("enabled", False))
        if draft_enabled:
            raise ValueError(
                "policy.draft.enabled=true is not supported on the DTensor backend yet. "
                "Disable policy.draft."
            )
        # Trainer-backend selection (mirrors the dtensor_cfg._v2 flag). Exactly
        # one of the JAX and DTensor backends is enabled.
        if jax_enable and dtensor_enable:
            raise ValueError(
                "policy.jax_cfg.enabled and policy.dtensor_cfg.enabled are mutually "
                "exclusive; enable exactly one trainer backend."
            )
        if not dtensor_enable and not jax_enable:
            raise ValueError(
                "Set policy.dtensor_cfg.enabled=true to use the DTensor training backend, "
                "or policy.jax_cfg.enabled=true for the JAX backend."
            )

        if jax_enable:
            from dockyard_rl.models.jax.config import resolve_trainer_env

            jax_cfg = config.get("jax_cfg", {})
            worker_builder_cls_fqn = "dockyard_rl.models.jax.policy_worker.JaxPolicyWorker"
            tp_size = int(jax_cfg.get("tensor_parallel_size", 1))
            cp_size = int(jax_cfg.get("context_parallel_size", 1))
            env_vars = resolve_trainer_env(jax_cfg)
        else:
            use_v2 = config.get("dtensor_cfg", {}).get("_v2", False)
            if use_v2:
                worker_builder_cls_fqn = resolve_policy_worker_cls(
                    "dockyard_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2",
                    config,
                )
                if "TORCH_CUDA_ARCH_LIST" not in os.environ:
                    warnings.warn(
                        "TORCH_CUDA_ARCH_LIST is not set. This is needed if using DeepEP in "
                        "DTensorPolicyWorker V2. Set it manually if required. "
                        "Example: export TORCH_CUDA_ARCH_LIST='9.0 10.0'"
                    )
            else:
                assert (
                    config["dtensor_cfg"].get("lora_cfg", {}).get("enabled", False)
                    is False
                ), "LoRA is not supported for DTensorPolicyWorker V1"
                worker_builder_cls_fqn = resolve_policy_worker_cls(
                    "dockyard_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker",
                    config,
                )

            tp_size = config["dtensor_cfg"]["tensor_parallel_size"]
            cp_size = config["dtensor_cfg"]["context_parallel_size"]

            env_vars = config["dtensor_cfg"].get("env_vars", {})

        # If a worker extension class is provided, use it instead of the default worker builder class
        if worker_extension_cls_fqn is not None:
            print(
                f"Using worker extension class: {worker_extension_cls_fqn}, please make sure it is a subclass of {worker_builder_cls_fqn}."
            )
            worker_builder_cls_fqn = worker_extension_cls_fqn

        # Validate world_size compatibility with parallelism configuration
        model_parallel_size = pp_size * cp_size * tp_size
        actual_world_size = cluster.world_size()

        if (
            not jax_enable
            and not bool(os.environ.get("DOCKYARD_IGNORE_TP_ACCURACY_CHECK"))
            and "logprob_batch_size" in config
            and tp_size >= 4
        ):
            sep_line = "\n" + ("-" * 80)
            assert config["train_micro_batch_size"] == config["logprob_batch_size"], (
                f"{sep_line}\n"
                "There is a known batch-variant accuracy issue with TP>=4 on the DTensor backend.\n"
                "\n"
                "Please choose either of the following solutions to avoid this issue:\n"
                "1. Set tp_size to 1 or 2 (policy.dtensor_cfg.tensor_parallel_size).\n"
                "2. Set policy.train_micro_batch_size and policy.logprob_batch_size to be the same value.\n"
                "3. Set loss_fn.force_on_policy_ratio=true to force ratio=1.0, this requires train_global_batch_size == num_prompts_per_step * num_generations_per_prompt.\n"
                "4. Set DOCKYARD_IGNORE_TP_ACCURACY_CHECK=1 to bypass this check. (not recommended)"
                f"{sep_line}\n"
            )

        if actual_world_size < model_parallel_size:
            raise ValueError(
                f"World size ({actual_world_size}) is insufficient for the parallelism configuration. "
                f"Required minimum world size: PP({pp_size}) * CP({cp_size}) * TP({tp_size}) = {model_parallel_size}. "
                f"This would result in DP = {actual_world_size}/{model_parallel_size} = {actual_world_size / model_parallel_size:.3f}, but DP must be ≥ 1. "
                f"Please either increase the number of GPUs/nodes or reduce the parallelism parameters."
            )

        if actual_world_size % model_parallel_size != 0:
            dp_size_float = actual_world_size / model_parallel_size
            raise ValueError(
                f"World size ({actual_world_size}) must be divisible by PP * CP * TP ({model_parallel_size}). "
                f"The data parallel size (DP = world_size / (PP * CP * TP)) must be a positive integer. "
                f"Current DP would be {actual_world_size}/{model_parallel_size} = {dp_size_float:.6f}, which is not an integer. "
                f"Please adjust your cluster size or parallelism parameters."
            )

        # MoE/EP config surface (v2 DTensor only): validate the expert-parallel
        # factoring + dispatcher selection early, before any device-mesh build,
        # so a bad config fails here with a clear message rather than as a cryptic
        # mesh/sharding error inside a worker.
        if use_v2:
            from dockyard_rl.models.dtensor.moe.config import (
                read_num_routed_experts,
                validate_moe_parallel_config,
            )

            dtensor_cfg = config["dtensor_cfg"]
            num_experts: Optional[int] = None
            if dtensor_cfg.get("expert_parallel_size", 1) > 1:
                num_experts = read_num_routed_experts(
                    config["model_name"], config.get("hf_config_overrides")
                )
                if num_experts is None:
                    raise ValueError(
                        "policy.dtensor_cfg.expert_parallel_size > 1 but the model "
                        f"config for '{config['model_name']}' exposes no recognized "
                        "routed-expert count (num_experts / n_routed_experts / "
                        "num_local_experts / moe_num_experts); expert parallelism "
                        "requires a MoE model."
                    )
            validate_moe_parallel_config(dtensor_cfg, actual_world_size, num_experts)

        self.sharding_annotations = NamedSharding(
            layout=np.arange(cluster.world_size()).reshape(
                pp_size,  # PP
                -1,  # DP
                cp_size,  # CP
                tp_size,  # TP
            ),
            names=[
                "pipeline_parallel",
                "data_parallel",
                "context_parallel",
                "tensor_parallel",
            ],
        )

        pre_init_queue = RayQueue()

        worker_kwargs: dict[str, Any] = dict(
            init_optimizer=init_optimizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_reference_model=init_reference_model,
            worker_sharding_annotations=self.sharding_annotations,
            pre_init_communication_queue=pre_init_queue,
        )

        if use_v2:
            # DTensor v2 workers reconstruct tokenizer/processor locally to avoid
            # pickling across incompatible transformers versions (v4 head → v5 worker).
            config["tokenizer"]["use_processor"] = processor is not None
        else:
            worker_kwargs["tokenizer"] = tokenizer
            worker_kwargs["processor"] = processor

        worker_builder = RayWorkerBuilder(
            worker_builder_cls_fqn,
            config,
            **worker_kwargs,
        )

        if cluster._sorted_bundle_indices is not None:
            # The cluster has initialized a unified placement group across nodes.
            # In this case, we need to create workers based on sorted bundle indices.
            group_size = cluster.num_gpus_per_node
            tied_groups = [
                (i // group_size, [bundle_idx])
                for i, bundle_idx in enumerate(cluster._sorted_bundle_indices)
            ]

            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                bundle_indices_list=tied_groups,
                sharding_annotations=self.sharding_annotations,
                env_vars=env_vars or {},
            )

        else:
            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                workers_per_node=workers_per_node,
                sharding_annotations=self.sharding_annotations,
                env_vars=env_vars or {},
            )

        if config["dynamic_batching"]["enabled"]:
            assert pp_size == 1, (
                "Dynamic batching is only supported for single pipeline parallel stage"
            )
            self.use_dynamic_batches = True
            self.dynamic_batching_args: DynamicBatchingArgs = {
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_round": config["dynamic_batching"][
                    "sequence_length_round"
                ],
                "max_tokens_per_microbatch": 0,  # Overridden at each call site
            }
            assert not config["sequence_packing"]["enabled"], (
                "Dynamic Batching is exclusive of Sequence Packing. Please disable Sequence Packing to use Dynamic Batching"
            )
        else:
            self.use_dynamic_batches = False

        if config["sequence_packing"]["enabled"]:
            self.use_sequence_packing = True
            sequence_length_pad_multiple = config["make_sequence_length_divisible_by"]
            self.sequence_packing_args: SequencePackingArgs = {
                "algorithm": config["sequence_packing"]["algorithm"],
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_pad_multiple": sequence_length_pad_multiple,
            }
            assert not config["dynamic_batching"]["enabled"], (
                "Sequence Packing is exclusive of Dynamic Batching. Please disable Dynamic Batching"
            )
        else:
            self.use_sequence_packing = False

        self.cfg = config

    def run_all_workers_single_data(self, method_name: str, *args, **kwargs) -> Any:
        """Run a method on all workers in parallel with the same data.

        Mainly used for worker extension classes.

        Args:
            method_name: The name of the method to run.
            *args: The positional arguments to pass to the method.
            **kwargs: The keyword arguments to pass to the method.

        Returns:
            The results of the method run on all workers.
        """
        futures = self.worker_group.run_all_workers_single_data(
            method_name, *args, **kwargs
        )
        results = ray.get(futures)
        return results

    def run_all_workers_multiple_data(self, method_name: str, *args, **kwargs) -> Any:
        """Run a method on all workers in parallel with different data.

        Mainly used for worker extension classes.

        Args:
            method_name: The name of the method to run.
            *args: The positional arguments to pass to the method.
            **kwargs: The keyword arguments to pass to the method.

        Returns:
            The results of the method run on all workers.
        """
        futures = self.worker_group.run_all_workers_multiple_data(
            method_name, *args, **kwargs
        )
        results = ray.get(futures)
        return results

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        """Initialize the collective communication."""
        futures = self.worker_group.run_all_workers_single_data(
            "init_collective",
            ip=ip,
            port=port,
            world_size=world_size,
            train_world_size=train_world_size,
        )
        # this function should co-work with vllm, so we should wait for all futures to complete outside
        return futures

    def get_logprobs(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        timer: Optional[Any] = None,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a data dict.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict]
        unsorted_data_indices: list[int] = []

        with timer.time("get_logprobs/shard_data") if timer else nullcontext():
            if self.use_dynamic_batches:
                self.dynamic_batching_args["max_tokens_per_microbatch"] = self.cfg[
                    "dynamic_batching"
                ]["logprob_mb_tokens"]
                sharded_data, unsorted_data_indices = data.shard_by_batch_size(  # type: ignore
                    dp_size,
                    batch_size=None,
                    dynamic_batching_args=self.dynamic_batching_args,
                )
            elif self.use_sequence_packing:
                self.sequence_packing_args["max_tokens_per_microbatch"] = self.cfg[
                    "sequence_packing"
                ]["logprob_mb_tokens"]
                sharded_data, unsorted_data_indices = data.shard_by_batch_size(
                    dp_size,
                    batch_size=None,
                    sequence_packing_args=self.sequence_packing_args,
                )
            else:
                sharded_data = data.shard_by_batch_size(  # type: ignore
                    dp_size,
                    batch_size=None,
                )

        with (
            timer.time("get_logprobs/submit_logprob_futures")
            if timer
            else nullcontext()
        ):
            futures = self.worker_group.run_all_workers_sharded_data(
                "get_logprobs",
                data=sharded_data,
                in_sharded_axes=["data_parallel"],
                replicate_on_axes=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
                output_is_replicated=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
            )
        logprobs: BatchedDataDict[LogprobOutputSpec] = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )

        # dynamic batching sorts the inputs by sequence length to improve load balancing,
        # so change it back here
        if self.use_dynamic_batches or self.use_sequence_packing:
            logprobs.reorder_data(unsorted_data_indices)

        return logprobs

    def get_reference_policy_logprobs(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
        timer: Optional[Any] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs of the reference policy for a data dict.

        Returns: Identical to get_logprobs.
        """
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict]
        unsorted_data_indices: list[int] = []
        with (
            timer.time("get_reference_policy_logprobs/shard_data")
            if timer
            else nullcontext()
        ):
            if self.use_dynamic_batches:
                self.dynamic_batching_args["max_tokens_per_microbatch"] = self.cfg[
                    "dynamic_batching"
                ]["logprob_mb_tokens"]
                sharded_data, unsorted_data_indices = data.shard_by_batch_size(  # type: ignore
                    dp_size,
                    batch_size=None,
                    dynamic_batching_args=self.dynamic_batching_args,
                )
            elif self.use_sequence_packing:
                self.sequence_packing_args["max_tokens_per_microbatch"] = self.cfg[
                    "sequence_packing"
                ]["logprob_mb_tokens"]
                sharded_data, unsorted_data_indices = data.shard_by_batch_size(
                    dp_size,
                    batch_size=None,
                    sequence_packing_args=self.sequence_packing_args,
                )
            else:
                sharded_data = data.shard_by_batch_size(  # type: ignore
                    dp_size,
                    batch_size=None,
                )

        with (
            timer.time(
                "get_reference_policy_logprobs/submit_reference_policy_logprob_futures"
            )
            if timer
            else nullcontext()
        ):
            futures = self.worker_group.run_all_workers_sharded_data(
                "get_reference_policy_logprobs",
                data=sharded_data,
                in_sharded_axes=["data_parallel"],
                replicate_on_axes=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
                output_is_replicated=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
                common_kwargs={"micro_batch_size": micro_batch_size},
            )
        logprobs: BatchedDataDict[ReferenceLogprobOutputSpec] = (
            BatchedDataDict.from_batches(
                self.worker_group.get_all_worker_results(futures)
            )
        )

        if self.use_dynamic_batches or self.use_sequence_packing:
            logprobs.reorder_data(unsorted_data_indices)

        return logprobs

    def get_topk_logits(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
        timer: Optional[Any] = None,
    ) -> BatchedDataDict[TopkLogitsOutputSpec]:
        """Dispatch get_topk_logits to workers."""
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict]
        unsorted_data_indices: list[int] = []
        with timer.time("get_topk_logits/shard_data") if timer else nullcontext():
            if self.use_dynamic_batches:
                self.dynamic_batching_args["max_tokens_per_microbatch"] = self.cfg[
                    "dynamic_batching"
                ]["logprob_mb_tokens"]
                sharded_data, unsorted_data_indices = data.shard_by_batch_size(  # type: ignore
                    dp_size,
                    batch_size=None,
                    dynamic_batching_args=self.dynamic_batching_args,
                )
            elif self.use_sequence_packing:
                self.sequence_packing_args["max_tokens_per_microbatch"] = self.cfg[
                    "sequence_packing"
                ]["logprob_mb_tokens"]
                sharded_data, unsorted_data_indices = data.shard_by_batch_size(
                    dp_size,
                    batch_size=None,
                    sequence_packing_args=self.sequence_packing_args,
                )
            else:
                sharded_data = data.shard_by_batch_size(  # type: ignore
                    dp_size,
                    batch_size=None,
                )

        with (
            timer.time("get_topk_logits/submit_topk_logits_futures")
            if timer
            else nullcontext()
        ):
            futures = self.worker_group.run_all_workers_sharded_data(
                "get_topk_logits",
                data=sharded_data,
                in_sharded_axes=["data_parallel"],
                replicate_on_axes=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
                output_is_replicated=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
                common_kwargs={"k": k, "micro_batch_size": micro_batch_size},
            )

        # Avoid BatchedDataDict.from_batches here because it flattens rows for tensors with ndim>2 ([B,S,k] -> [B,S*k]).
        worker_batches = self.worker_group.get_all_worker_results(futures)
        all_topk_logits = [wb["topk_logits"] for wb in worker_batches]
        all_topk_indices = [wb["topk_indices"] for wb in worker_batches]

        stacked: BatchedDataDict[TopkLogitsOutputSpec] = BatchedDataDict()
        stacked["topk_logits"] = torch.cat(all_topk_logits, dim=0)
        stacked["topk_indices"] = torch.cat(all_topk_indices, dim=0)

        if self.use_dynamic_batches or self.use_sequence_packing:
            stacked.reorder_data(unsorted_data_indices)

        return stacked

    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        timer: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        batch_size = gbs or self.cfg["train_global_batch_size"]
        micro_batch_size = mbs or self.cfg["train_micro_batch_size"]
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        with timer.time("policy_training/sharding_data") if timer else nullcontext():
            if self.use_dynamic_batches:
                self.dynamic_batching_args["max_tokens_per_microbatch"] = self.cfg[
                    "dynamic_batching"
                ]["train_mb_tokens"]
                sharded_data, _ = data.shard_by_batch_size(
                    dp_size,
                    batch_size=batch_size,
                    dynamic_batching_args=self.dynamic_batching_args,
                )
            elif self.use_sequence_packing:
                self.sequence_packing_args["max_tokens_per_microbatch"] = self.cfg[
                    "sequence_packing"
                ]["train_mb_tokens"]
                sharded_data, _ = data.shard_by_batch_size(
                    dp_size,
                    batch_size=batch_size,
                    sequence_packing_args=self.sequence_packing_args,
                )
            else:
                sharded_data = data.shard_by_batch_size(
                    dp_size,
                    batch_size=batch_size,
                )

        with (
            timer.time("policy_training/submit_training_futures")
            if timer
            else nullcontext()
        ):
            futures = self.worker_group.run_all_workers_sharded_data(
                "train",
                data=sharded_data,
                in_sharded_axes=["data_parallel"],
                replicate_on_axes=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
                output_is_replicated=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
                common_kwargs={
                    "loss_fn": loss_fn,
                    "eval_mode": eval_mode,
                    "gbs": batch_size,
                    "mbs": micro_batch_size,
                },
            )
        results = self.worker_group.get_all_worker_results(futures)

        # Aggregate the results
        aggregated_results = {
            "loss": results[0]["global_loss"],
            "grad_norm": results[0]["grad_norm"],
        }
        if "moe_metrics" in results[0]:
            aggregated_results["moe_metrics"] = results[0]["moe_metrics"]

        # Aggregate metrics across all workers
        all_mb_metrics = defaultdict(list)
        for r in results:
            for k, v in r["all_mb_metrics"].items():
                all_mb_metrics[k].extend(v)
        aggregated_results["all_mb_metrics"] = dict(all_mb_metrics)

        return aggregated_results

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using the policy."""
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "Missing required input fields"
        )

        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data = data.shard_by_batch_size(dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_sharded_data(
            "generate",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["tensor_parallel", "pipeline_parallel"],
            output_is_replicated=["tensor_parallel", "pipeline_parallel"],
            common_kwargs={"greedy": greedy},
        )
        assert self.cfg["generation"] is not None, "Generation config is not set"
        result: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures),
            pad_value_dict={"output_ids": self.cfg["generation"]["_pad_token_id"]},
        )

        required_keys = [
            "output_ids",
            "generation_lengths",
            "unpadded_sequence_lengths",
            "logprobs",
        ]
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for GenerationOutputSpec: {missing_keys}"
            )

        return result

    def score(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict[ScoreOutputSpec]:
        """Score a batch of data using the policy."""
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "Missing required input fields"
        )

        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data = data.shard_by_batch_size(dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_sharded_data(
            "score",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            output_is_replicated=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            common_kwargs={},
        )

        result: BatchedDataDict[ScoreOutputSpec] = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures),
        )
        missing_keys = [k for k in ("scores",) if k not in result]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for ScoreOutputSpec: {missing_keys}"
            )

        return result

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        futures = self.worker_group.run_all_workers_single_data("prepare_for_training")
        ray.get(futures)

    def prepare_for_lp_inference(self, *args: Any, **kwargs: Any) -> None:
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_for_lp_inference"
        )
        ray.get(futures)

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def invalidate_kv_cache(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def prepare_refit_info(self, state_dict_info: Optional[Any] = None) -> Optional[dict[str, Any]]:  # type: ignore[override]
        """Prepare the info for refit.

        Returns:
            dict: A dictionary containing the info for refit.
        """
        futures = self.worker_group.run_all_workers_single_data("prepare_refit_info")
        results = ray.get(futures)
        return results[0]

    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        pass

    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> list[ray.ObjectRef]:
        """Broadcast the weights for collective communication."""
        futures = self.worker_group.run_all_workers_single_data(
            "broadcast_weights_for_collective",
            kv_scales=kv_scales,
        )
        return futures

    def stream_weights_via_http(
        self, sglang_url_to_gpu_uuids: dict[str, list[str]]
    ) -> list[ray.ObjectRef]:
        """Stream weights to SGLang servers over HTTP from every worker."""
        futures = self.worker_group.run_all_workers_single_data(
            "stream_weights_via_http",
            sglang_url_to_gpu_uuids=sglang_url_to_gpu_uuids,
        )
        return futures

    def offload_before_refit(self) -> None:
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data("offload_before_refit")
        ray.get(futures)

    def offload_after_refit(self) -> None:
        """Offload as much as possible to the CPU."""
        futures = self.worker_group.run_all_workers_single_data("offload_after_refit")
        ray.get(futures)

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        checkpointing_cfg: Optional[Any] = None,
    ) -> None:
        """Save a checkpoint of the model."""
        use_v2 = self.cfg.get("dtensor_cfg", {}).get("_v2", False)

        if use_v2:
            futures = self.worker_group.run_all_workers_single_data(
                "save_checkpoint",
                weights_path=weights_path,
                optimizer_path=optimizer_path,
                tokenizer_path=tokenizer_path,
                checkpointing_cfg=checkpointing_cfg,
            )
        else:
            if (
                checkpointing_cfg is not None
                and checkpointing_cfg.get("model_save_format", None) is not None
            ):
                raise ValueError(
                    "model_save_format must be None or omitted if using DTensorPolicyWorker (_v2=False)."
                )
            futures = self.worker_group.run_all_workers_single_data(
                "save_checkpoint",
                weights_path=weights_path,
                optimizer_path=optimizer_path,
                tokenizer_path=tokenizer_path,
            )
        ray.get(futures)

    def get_free_memory_bytes(self) -> int:
        """Get the available free memory (minimum across all workers)."""
        futures = self.worker_group.run_all_workers_single_data("get_free_memory_bytes")
        return min(ray.get(future) for future in futures)

    def shutdown(self) -> bool:
        """Shut down all workers and clean up resources."""
        try:
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during policy shutdown: {e}")
            return False

    def __del__(self) -> None:
        """Extra safety net to shut down workers if the object is garbage collected."""
        if hasattr(self, "worker_group"):
            self.worker_group.shutdown(cleanup_method="shutdown")

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        futures = self.worker_group.run_all_workers_single_data("start_gpu_profiling")
        ray.get(futures)

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        futures = self.worker_group.run_all_workers_single_data("stop_gpu_profiling")
        ray.get(futures)

    def print_node_ip_and_gpu_id(self) -> list[tuple[str, int]]:
        """Print the node IP and GPU ID of the current worker."""
        results = ray.get(
            self.worker_group.run_all_workers_single_data(
                "report_node_ip_and_gpu_id",
            )
        )
        all_node_ips = sorted(set([result[0] for result in results]))
        all_gpu_ids = sorted(set([result[1] for result in results]))

        worker_id_list = [
            [list() for _ in range(len(all_gpu_ids))] for _ in range(len(all_node_ips))
        ]
        for worker_id, (ip, gpu_id) in enumerate(results):
            node_idx = all_node_ips.index(ip)
            gpu_idx = all_gpu_ids.index(gpu_id)
            worker_id_list[node_idx][gpu_idx].append("worker-" + str(worker_id))

        from prettytable import PrettyTable

        table = PrettyTable()
        table.title = "Policy worker mapping to Nodes and GPUs"
        table.field_names = ["Node_IP"] + [
            "GPU_ID=" + str(gpu_id) for gpu_id in all_gpu_ids
        ]
        for i, node_idx in enumerate(all_node_ips):
            row = [node_idx]
            for j in range(len(all_gpu_ids)):
                row.append(tuple(worker_id_list[i][j]))
            table.add_row(row)

        print(table)
        return [(ip, gpu_id) for ip, gpu_id in results]