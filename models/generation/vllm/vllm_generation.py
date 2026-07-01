"""VllmGeneration: inference fleet manager for Project Dockyard.

Wraps a RayWorkerGroup of VllmGenerationWorker (or VllmAsyncGenerationWorker)
actors and exposes the GenerationInterface consumed by the GRPO training loop's
weight-refit path (dockyard_rl.weight_sync WeightSynchronizer transports).

Architecture
------------
Non-collocated only.  The inference fleet runs as a dedicated RayWorkerGroup
on GPU nodes separate from the trainer.  Weight sync is via NCCL collective.
Collocated mode (wake_up / sleep / IPC ZMQ) has been removed.

DP semantics
------------
Each DP shard is a tied worker group: one group of TP×PP workers that jointly
hold one complete model replica.  VllmGeneration splits incoming batches
across DP shards, dispatches to each tied group leader, and merges results.

Parallelism
-----------
- TP ≤ gpus_per_node     → per-node PACK placement groups.
- TP > gpus_per_node     → unified STRICT_PACK group spanning all nodes;
                           detected via placement.requires_unified_placement_group().
- EP > 1 (MoE)           → expert-parallel env vars set in worker configure_worker().
"""

import asyncio
import logging
import os
import time
from typing import Any, AsyncGenerator
import ray
import torch
from dockyard_rl.cluster.placement import (
    ParallelismSpec,
    requires_unified_placement_group,
)
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.distributed.virtual_cluster import RayVirtualCluster
from dockyard_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from dockyard_rl.models.generation.interfaces import (
    GenerationInterface,
    GenerationDatumSpec,
    GenerationOutputSpec,
)
from dockyard_rl.models.generation.vllm.config import VllmConfig
from dockyard_rl.models.generation.vllm.router_capture import (
    MISSING_ROUTE_SENTINEL,
)
from dockyard_rl.models.generation.vllm.utils import (
    aggregate_spec_decode_counters,
    compute_spec_decode_metrics,
    resolve_generation_worker_cls,
)

logger = logging.getLogger(__name__)

class VllmGeneration(GenerationInterface):
    """Manages the vLLM inference fleet as a RayWorkerGroup.

    Args:
        config:      VllmConfig — see models/generation/vllm/config.py.
        cluster:     RayVirtualCluster for the inference fleet.  Must have
                     been initialised via fleet.build_cluster(inference_spec).
        parallelism: ParallelismSpec for the inference fleet.  Used to
                     detect cross-node TP and set VLLM_DP_SIZE.

    Usage:
        gen = VllmGeneration(config, cluster, parallelism)
        gen.init_collective(ip, port, world_size, train_world_size=T)
        outputs = gen.generate(data, greedy=False)
        gen.update_weights_from_collective()
    """

    def __init__(
        self,
        config:      VllmConfig,
        cluster:     RayVirtualCluster,
        parallelism: ParallelismSpec,
    ) -> None:
        self.cfg         = config
        self.cluster     = cluster
        self.parallelism = parallelism

        self._dp_size  = parallelism.data_parallel_size
        self._tp_size  = parallelism.tensor_parallel_size
        self._pp_size  = parallelism.pipeline_parallel_size
        self._ep_size  = parallelism.expert_parallel_size
        self._model_parallel_size = self._tp_size * self._pp_size

        # Round-robin cursor for assigning single-sample async generations to
        # DP-leader workers (used by generate_async).
        self.current_generate_dp_shard_idx = 0

        # Resolve worker class.
        # quant_cfg present → route to ModelOpt-backed quantized worker.
        # Otherwise → standard worker (sync or async based on async_engine).
        use_async = self.cfg["vllm_cfg"].get("async_engine", False)
        quant_cfg = self.cfg.get("quant_cfg")

        if quant_cfg:
            default_fqn = (
                "dockyard_rl.modelopt.models.generation.vllm_quant_worker"
                ".VllmQuantAsyncGenerationWorker"
                if use_async
                else
                "dockyard_rl.modelopt.models.generation.vllm_quant_worker"
                ".VllmQuantGenerationWorker"
            )
        else:
            default_fqn = (
                "dockyard_rl.models.generation.vllm.vllm_worker_async"
                ".VllmAsyncGenerationWorker"
                if use_async
                else
                "dockyard_rl.models.generation.vllm.vllm_worker"
                ".VllmGenerationWorker"
            )
        worker_cls_fqn = resolve_generation_worker_cls(default_fqn, config)

        # Build bundle indices list for worker group constructor
        # Each entry is (pg_idx, [local_bundle_indices]) defining one
        # tied worker group (one DP shard's TP×PP ranks on one node).
        cross_node_tp = requires_unified_placement_group(
            parallelism,
            cluster.num_gpus_per_node,
        )
        bundle_indices_list = self._build_bundle_indices(cross_node_tp)

        # Expert-parallel env var
        env_vars: dict[str, str] = {}
        if self._ep_size > 1:
            env_vars["VLLM_DP_SIZE"] = str(self._dp_size)
        # Per-recipe env vars (e.g. fused-MoE backend selection), scoped to this config.
        for k, v in self.cfg["vllm_cfg"].get("env_vars", {}).items():
            env_vars[str(k)] = str(v)

        # Initialise placement groups
        cluster._init_placement_groups(
            use_unified_pg=cross_node_tp
        )

        # Construct worker group
        self.worker_group = RayWorkerGroup(
            cluster=cluster,
            remote_worker_builder=RayWorkerBuilder(worker_cls_fqn, config),
            bundle_indices_list=bundle_indices_list,
            name_prefix="dockyard-inference",
            env_vars=env_vars,
        )

        # Post-init: warm up each DP leader with a no-op call to populate CUDA context and report device info.
        post_init_futures = self.worker_group.run_all_workers_single_data(
            "post_init",
            run_rank_0_only_axes=None,
        )
        try:
            ray.get(post_init_futures, timeout=300)
        except Exception as exc:
            raise RuntimeError(
                "vLLM worker post_init() timed out or failed. "
                "Check GPU memory on inference nodes."
            ) from exc

        logger.info(
            "VllmGeneration ready: dp=%d tp=%d pp=%d ep=%d",
            self._dp_size, self._tp_size, self._pp_size, self._ep_size,
        )

        # Spec-decode metric state.
        self._spec_start_counters: dict = {}
        self._last_gen_metrics: dict[str, float] = {}

    # GenerationInterface: collective init and generation
    def init_collective(
        self,
        ip:               str,
        port:             int,
        world_size:       int,
        *,
        train_world_size: int,
    ) -> list[ray.ObjectRef]:
        """Initialise NCCL weight-sync communicator on all DP leaders.

        Rank assignment inside the sync communicator:
          ranks 0 … T-1       ← trainer DP leaders
          ranks T … T+I-1     ← inference DP leaders  (T = train_world_size)

        Each inference DP leader joins at rank T + shard_idx.
        """
        futures = []
        for shard_idx, worker_idx in enumerate(
            self.worker_group.dp_leader_worker_indices
        ):
            rank_in_sync = train_world_size + shard_idx
            futures.append(
                self.worker_group.run_single_worker_single_data(
                    "init_collective",
                    worker_idx=worker_idx,
                    rank_prefix=rank_in_sync,
                    ip=ip,
                    port=port,
                    world_size=world_size,
                    train_world_size=train_world_size,
                )
            )
        return futures

    # GenerationInterface: generate
    def generate(
        self,
        data:   BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Batch generation across DP shards.

        Splits the batch evenly across DP workers, dispatches generation
        in parallel, and merges results (padding to a common sequence
        length where shards may produce different-length outputs).

        Args:
            data:   Input BatchedDataDict with input_ids and input_lengths.
            greedy: If True, use greedy decoding (temperature=0, top_k=1).

        Returns:
            Merged BatchedDataDict with output_ids, logprobs, etc.
        """
        t0 = time.perf_counter()

        if self._dp_size == 1:
            worker_idx = self.worker_group.dp_leader_worker_indices[0]
            result = ray.get(
                self.worker_group.run_single_worker_single_data(
                    "generate",
                    worker_idx=worker_idx,
                    data=data,
                    greedy=greedy,
                )
            )
            self._last_gen_metrics["generation_time_s"] = (
                time.perf_counter() - t0
            )
            return result

        # Shard across DP workers.
        shards = data.shard_by_batch_size(
            self._dp_size,
            allow_uneven_shards=True,
        )
        futures = []
        for shard_idx, (shard, worker_idx) in enumerate(
            zip(shards, self.worker_group.dp_leader_worker_indices)
        ):
            if shard._batch_size is None or shard._batch_size == 0:
                # Empty shard — still dispatch a no-op so workers stay in sync.
                futures.append(
                    self.worker_group.run_single_worker_single_data(
                        "generate",
                        worker_idx=worker_idx,
                        data=BatchedDataDict({
                            "input_ids":     torch.zeros(
                                (0, data["input_ids"].shape[1]),
                                dtype=data["input_ids"].dtype,
                            ),
                            "input_lengths": torch.zeros(0, dtype=torch.long),
                        }),
                        greedy=greedy,
                    )
                )
            else:
                futures.append(
                    self.worker_group.run_single_worker_single_data(
                        "generate",
                        worker_idx=worker_idx,
                        data=shard,
                        greedy=greedy,
                    )
                )

        shard_results: list[BatchedDataDict[GenerationOutputSpec]] = ray.get(futures)
        merged = BatchedDataDict.from_batches(
            shard_results,
            pad_value_dict={
                "output_ids":     self.cfg.get("_pad_token_id", 0),
                "logprobs":       0.0,
                # Cross-shard sequence padding for the MoE router-replay column
                # (#2908): pad positions get the missing-route sentinel so the
                # trainer replays its own routing there. Inert when the column is
                # absent (router_replay disabled / non-MoE).
                "routed_experts": MISSING_ROUTE_SENTINEL,
            },
        )

        self._last_gen_metrics["generation_time_s"] = time.perf_counter() - t0
        return merged

    async def _async_generate_base(
        self,
        data:                BatchedDataDict[GenerationDatumSpec],
        method_name:         str,
        data_validation_fn,
        greedy:              bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Drive a single-sample async generation on one DP-leader worker.

        Routes the sample to a leader worker round-robin, awaits the streamed
        result ref, materializes it with a timeout, and yields
        (original_idx, result_batch). Per-sample concurrency is the caller's
        responsibility (the rollout layer fires many of these in parallel).
        """
        if not self.cfg["vllm_cfg"].get("async_engine", False):
            raise RuntimeError(
                f"{method_name} can only be used when async_engine is enabled "
                "in the vLLM config."
            )
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        if not data_validation_fn(data):
            return
        batch_size = len(data["input_ids"])
        assert batch_size == 1, (
            f"{method_name} is restricted to single samples, but received "
            f"batch_size={batch_size}. Batch outside this method."
        )

        leader_worker_idx = self.worker_group.get_dp_leader_worker_idx(
            self.current_generate_dp_shard_idx
        )
        # run_single_worker_single_data is typed -> ObjectRef, but an
        # async-generator worker method returns an ObjectRefGenerator that
        # supports __anext__; treat it as such at the boundary.
        worker_gen_proxy: Any = self.worker_group.run_single_worker_single_data(
            method_name=method_name,
            worker_idx=leader_worker_idx,
            data=data,
            greedy=greedy,
        )
        self.current_generate_dp_shard_idx = (
            self.current_generate_dp_shard_idx + 1
        ) % self.worker_group.dp_size

        timeout_seconds = float(
            os.environ.get("DOCKYARD_VLLM_ASYNC_TIMEOUT_SECONDS", "900")
        )
        try:
            sample_result_ref = await anext(worker_gen_proxy)
        except StopAsyncIteration:
            raise RuntimeError(f"Worker produced no output for sample {data}.")

        # anext resolves when the worker yields, but the bytes have not yet
        # crossed to the driver — awaiting the ref is where that happens and
        # where an unreachable worker would hang, hence the timeout.
        try:
            sample_result = await asyncio.wait_for(
                sample_result_ref, timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Timeout waiting for worker results after {timeout_seconds}s. "
                "Increase via DOCKYARD_VLLM_ASYNC_TIMEOUT_SECONDS="
                f"{int(timeout_seconds * 2)}"
            )

        original_idx, result_batch = sample_result
        result_batch["gen_leader_worker_idx"] = [int(leader_worker_idx)]
        yield (original_idx, result_batch)

    async def generate_async(
        self,
        data:   BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Per-sample streaming generation; yields (original_idx, result) as ready."""

        def _validate(d: BatchedDataDict[GenerationDatumSpec]) -> bool:
            if "input_ids" not in d or "input_lengths" not in d:
                raise AssertionError(
                    "input_ids and input_lengths are required for vLLM generation"
                )
            return len(d["input_ids"]) != 0

        async for result in self._async_generate_base(
            data, "generate_async", _validate, greedy
        ):
            yield result

    # GenerationInterface: hooks for training loop integration
    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """No-op for non-collocated: vLLM is always-on."""
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Invalidate prefix cache at end of rollout batch."""
        return self.invalidate_kv_cache()

    def invalidate_kv_cache(self) -> bool:
        """Reset vLLM's prefix/KV cache on all DP leaders.

        Must be called after each weight update to prevent stale KV blocks
        computed with the old policy from being reused.
        """
        method = (
            "reset_prefix_cache_async"
            if self.cfg["vllm_cfg"].get("async_engine", False)
            else "reset_prefix_cache"
        )
        futures = self.worker_group.run_all_workers_single_data(
            method,
            run_rank_0_only_axes=None,
        )
        try:
            results = ray.get(futures, timeout=60)
            return all(r is not False for r in results)
        except Exception as exc:
            logger.warning("KV cache invalidation failed: %s", exc)
            return False

    # GenerationInterface: weight sync
    def prepare_refit_info(
        self,
        state_dict_info: dict[str, Any],
    ) -> None:
        """Forward weight metadata to all DP leaders before NCCL broadcast."""
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_refit_info",
            run_rank_0_only_axes=None,
            state_dict_info=state_dict_info,
        )
        ray.get(futures)

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        """Trigger NCCL weight receive + refit on all DP leaders.

        Returns Ray futures — callers should ray.get() them alongside the
        trainer-side broadcast futures so both sides block until complete.
        """
        return self.worker_group.run_all_workers_single_data(
            "update_weights_from_collective",
            run_rank_0_only_axes=None,
        )

    # Metrics and logging
    def clear_logger_metrics(self) -> None:
        """Snapshot spec-decode counters at the start of a training step."""
        futures = self.worker_group.run_all_workers_single_data(
            "_get_raw_spec_counters",
            run_rank_0_only_axes=None,
        )
        try:
            per_worker = ray.get(futures, timeout=10)
            self._spec_start_counters = aggregate_spec_decode_counters(
                per_worker
            )
        except Exception:
            self._spec_start_counters = {}

    def get_logger_metrics(self) -> dict[str, float]:
        """Return telemetry metrics for this training step."""
        futures = self.worker_group.run_all_workers_single_data(
            "_get_raw_spec_counters",
            run_rank_0_only_axes=None,
        )
        try:
            per_worker = ray.get(futures, timeout=10)
            end_counters = aggregate_spec_decode_counters(per_worker)
            spec_metrics = compute_spec_decode_metrics(
                self._spec_start_counters, end_counters
            )
        except Exception:
            spec_metrics = {}

        return {**self._last_gen_metrics, **spec_metrics}

    # Shutdown
    def shutdown(self) -> bool:
        """Shutdown all inference workers."""
        return self.worker_group.shutdown(
            cleanup_method="shutdown",
            timeout=60,
        )

    # Internal utilities
    def _build_bundle_indices(
        self,
        cross_node_tp: bool,
    ) -> list[tuple[int, list[int]]]:
        """Build the bundle_indices_list for the worker group constructor.

        Each entry defines a tied worker group: the pg_idx and local bundle
        indices that collectively hold one DP shard's TP×PP ranks.

        For per-node topology (TP ≤ gpus_per_node):
          Each placement group (one per node) contains TP×PP bundles.
          One tied group per placement group.

        For unified topology (cross-node TP):
          One placement group spans all nodes.
          Bundles are grouped by model_parallel_size (TP×PP) consecutively.
        """
        cluster_pgs = self.cluster.get_placement_groups()
        sorted_indices = self.cluster._sorted_bundle_indices

        if cross_node_tp:
            # Unified PG: group consecutive bundles by model_parallel_size.
            assert len(cluster_pgs) == 1, (
                "Cross-node TP requires a single unified placement group."
            )
            # Use world_size() — authoritative total bundle count, no
            # dependency on Ray's internal bundle_specs attribute.
            n_bundles = self.cluster.world_size()
            indices   = sorted_indices if sorted_indices else list(range(n_bundles))
            result: list[tuple[int, list[int]]] = []
            for start in range(0, len(indices), self._model_parallel_size):
                chunk = indices[start: start + self._model_parallel_size]
                if len(chunk) == self._model_parallel_size:
                    result.append((0, chunk))
            return result

        # Per-node: one tied group per placement group.
        # Bundles per node = total_world_size / num_nodes — avoids bundle_specs.
        num_nodes      = self.cluster.node_count()
        bundles_per_pg = self.cluster.world_size() // max(num_nodes, 1)
        result = []
        for pg_idx in range(len(cluster_pgs)):
            local_indices = list(range(bundles_per_pg))
            result.append((pg_idx, local_indices))
        return result