import asyncio
import logging
import os
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Optional,
    Union,
)

import numpy as np
import ray

from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.distributed.named_sharding import NamedSharding
from dockyard_rl.distributed.virtual_cluster import RayVirtualCluster
from dockyard_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from dockyard_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from dockyard_rl.models.generation.sglang.config import SGLangConfig

if TYPE_CHECKING:
    # SlicedDataDict is only referenced in a (never-evaluated) local annotation;
    # import it for the type checker without forcing a runtime dependency.
    from dockyard_rl.distributed.batched_data_dict import SlicedDataDict

# Global thresholds for top_k and top_p validation.
# While top-k/p are not supported, these values allow for token filtering while the logprobs should be compatible.
TOP_K_THRESHOLD = 8000  # Allow top_k >= 8000 (effectively no filtering)
TOP_P_THRESHOLD = 0.99  # Allow top_p >= 0.99 (close to 1.0)

logger = logging.getLogger(__name__)


class SGLangGeneration(GenerationInterface):
    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: SGLangConfig,
        name_prefix: str = "sglang_policy",
        workers_per_node: Optional[Union[int, list[int]]] = None,
    ):
        """Initialize a SGLang policy with distributed workers.

        SGLang server manages TP/PP internally, but we still need to:
        1. Manage data parallel distribution across multiple servers
        2. Assign GPU bundles to each server

        Each server will see logical GPUs 0-N (via CUDA_VISIBLE_DEVICES set by Ray),
        so we just need to tell SGLang how many GPUs to use (tp_size).
        """
        # Store config
        self.cfg = config
        self.sglang_cfg = config["sglang_cfg"]

        gpus_per_server = self.sglang_cfg.get("gpus_per_server", None)
        if gpus_per_server is None:
            raise ValueError("gpus_per_server must be set in SGLangConfig.sglang_cfg.")

        # Calculate number of servers based on available resources
        total_gpus = cluster.world_size()
        num_servers = total_gpus // gpus_per_server

        if num_servers == 0:
            raise ValueError(
                f"Not enough GPUs. Need at least {gpus_per_server} GPUs per server, "
                f"but only have {total_gpus} GPUs total."
            )

        if total_gpus % gpus_per_server != 0:
            logger.warning(
                f"[WARNING] Total GPUs ({total_gpus}) is not divisible by GPUs per server ({gpus_per_server}). "
                f"Will use {num_servers} servers, leaving {total_gpus % gpus_per_server} GPUs unused."
            )

        self.dp_size = num_servers
        self.gpus_per_server = gpus_per_server

        # Create sharding annotations
        # Even though SGLang manages TP internally, we include it in the layout to support
        # RayWorkerGroup's worker management (which creates one worker per GPU bundle).
        # The TP dimension becomes a "free axis" in run_all_workers_sharded_data, ensuring
        # only the primary workers (TP rank 0) are called.
        total_workers = num_servers * gpus_per_server
        self.sharding_annotations = NamedSharding(
            layout=np.arange(total_workers).reshape(num_servers, gpus_per_server),
            names=["data_parallel", "tensor_parallel"],
        )

        # Initialize placement groups.
        # Dockyard inference workers are always non-collocated (dedicated fleet),
        # so per-node bundles use the STRICT_PACK strategy for NVLink locality.
        cluster._init_placement_groups(
            strategy="STRICT_PACK",
            use_unified_pg=False,  # SGLang servers don't need cross-node model parallelism
        )

        # Create worker builder for SGLangGenerationWorker
        worker_cls = (
            "dockyard_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker"
        )
        worker_builder = RayWorkerBuilder(worker_cls, config)

        env_vars = {}
        global_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if global_cvd:
            # Explicitly pass CUDA_VISIBLE_DEVICES to workers via env_vars
            # This ensures all workers see the same global value, even though
            env_vars["CUDA_VISIBLE_DEVICES"] = global_cvd

        # Allocate bundles for each server
        # Each server gets consecutive bundles
        bundle_indices_list = self._allocate_bundles_for_servers(
            cluster, num_servers, gpus_per_server
        )

        # Create worker group with explicit bundle allocation
        self.worker_group = RayWorkerGroup(
            cluster,
            worker_builder,
            name_prefix=name_prefix,
            bundle_indices_list=bundle_indices_list,
            sharding_annotations=self.sharding_annotations,
            env_vars=env_vars,
        )

        # Verify data parallel size matches
        assert self.dp_size == self.worker_group.dp_size, (
            f"Data parallel size mismatch. Expected {self.dp_size}, got {self.worker_group.dp_size}"
        )

        # Used to track the round-robin selection of worker groups for generate_async
        self.current_generate_dp_shard_idx = 0

    def _allocate_bundles_for_servers(
        self,
        cluster: RayVirtualCluster,
        num_servers: int,
        gpus_per_server: int,
    ) -> list[tuple[int, list[int]]]:
        """Allocate GPU bundles to each SGLang server.

        Each server gets consecutive bundles within the same placement group (node).
        Ray will automatically set CUDA_VISIBLE_DEVICES so each server sees logical GPUs 0, 1, 2, ..., gpus_per_server-1.

        Args:
            cluster: The Ray virtual cluster
            num_servers: Total number of SGLang servers to create
            gpus_per_server: Number of GPUs each server needs

        Returns:
            List of (node_idx, [bundle_indices]) tuples for each server
        """
        placement_groups = cluster.get_placement_groups()

        if not placement_groups:
            raise ValueError("No placement groups available in the cluster")

        bundle_indices_list = []

        # Each server's bundles must be within the same placement group (node)
        server_idx = 0
        for pg_idx, pg in enumerate(placement_groups):
            if pg.bundle_count == 0:
                continue

            # Calculate how many servers can fit in this placement group
            num_servers_in_pg = pg.bundle_count // gpus_per_server

            # Allocate servers within this placement group
            for local_server_idx in range(num_servers_in_pg):
                if server_idx >= num_servers:
                    break

                # Calculate which bundles this server gets (consecutive within the PG)
                start_bundle = local_server_idx * gpus_per_server
                server_bundles = list(
                    range(start_bundle, start_bundle + gpus_per_server)
                )

                # Each server gets a tuple of (node_idx, [local_bundle_indices])
                bundle_indices_list.append((pg_idx, server_bundles))
                server_idx += 1

            if server_idx >= num_servers:
                break

        if len(bundle_indices_list) < num_servers:
            total_available = sum(
                pg.bundle_count // gpus_per_server
                for pg in placement_groups
                if pg.bundle_count > 0
            )
            raise ValueError(
                f"Not enough bundles to allocate all {num_servers} servers. "
                f"Only {total_available} servers can be allocated "
                f"(each server needs {gpus_per_server} GPUs)."
            )

        return bundle_indices_list

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        """No-op for the SGLang backend.

        SGLang receives refreshed weights over its HTTP ``/update_weights_from_tensor``
        endpoint using CUDA IPC handles gathered on the trainer side (see
        ``stream_weights_via_http_impl`` in ``models/policy/utils.py``). That path is
        self-contained and needs no pre-established NCCL weight-sync communicator, so
        there is nothing to initialise here. The method is required by
        ``GenerationInterface``; it intentionally starts no collective and returns an
        empty list of object refs (no remote work is scheduled).
        """
        return []

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using SGLang."""
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "input_ids and input_lengths are required in data for SGLang generation"
        )

        # Shard the data across the data parallel servers
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict] = data.shard_by_batch_size(
            dp_size, allow_uneven_shards=True
        )
        future_bundle = self.worker_group.run_all_workers_sharded_data(
            "generate",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=None,
            output_is_replicated=None,
            common_kwargs={"greedy": greedy},
        )

        # Get results from the workers
        results = self.worker_group.get_all_worker_results(future_bundle)

        # Combine results from all servers
        combined: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict.from_batches(
            results, pad_value_dict={"output_ids": self.cfg.get("_pad_token_id", 0)}
        )

        # Verify the output has all required fields
        required_keys = [
            "output_ids",
            "generation_lengths",
            "unpadded_sequence_lengths",
            "logprobs",
        ]
        missing_keys = [key for key in required_keys if key not in combined]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for GenerationOutputSpec: {missing_keys}"
            )

        return combined

    async def _async_generate_base(
        self,
        data:        BatchedDataDict[GenerationDatumSpec],
        method_name: str,
        greedy:      bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Route one sample to a DP-leader server (round-robin) and stream it.

        Per-sample concurrency is the caller's responsibility; the worker runs
        as a concurrent async actor so many of these execute in parallel.
        """
        if not self.cfg.get("sglang_cfg", {}).get("async_engine", False):
            raise RuntimeError(
                f"{method_name} requires sglang_cfg.async_engine=True."
            )
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        if "input_ids" not in data or "input_lengths" not in data:
            raise AssertionError(
                "input_ids and input_lengths are required for SGLang generation"
            )
        if len(data["input_ids"]) == 0:
            return
        batch_size = len(data["input_ids"])
        assert batch_size == 1, (
            f"{method_name} is restricted to single samples, but received "
            f"batch_size={batch_size}. Batch outside this method."
        )

        leader_worker_idx = self.worker_group.get_dp_leader_worker_idx(
            self.current_generate_dp_shard_idx
        )
        # An async-generator worker method returns an ObjectRefGenerator that
        # supports __anext__; run_single_worker_single_data is typed -> ObjectRef.
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
            os.environ.get("DOCKYARD_SGLANG_ASYNC_TIMEOUT_SECONDS", "900")
        )
        try:
            sample_result_ref = await anext(worker_gen_proxy)
        except StopAsyncIteration:
            raise RuntimeError(f"Worker produced no output for sample {data}.")

        try:
            sample_result = await asyncio.wait_for(
                sample_result_ref, timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Timeout waiting for SGLang worker results after {timeout_seconds}s. "
                "Increase via DOCKYARD_SGLANG_ASYNC_TIMEOUT_SECONDS="
                f"{int(timeout_seconds * 2)}"
            )

        original_idx, result_batch = sample_result
        result_batch["gen_leader_worker_idx"] = [int(leader_worker_idx)]
        yield (original_idx, result_batch)

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Per-sample streaming generation; yields (original_idx, result) as ready."""
        async for result in self._async_generate_base(data, "generate_async", greedy):
            yield result

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        pass

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        return []

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        return []

    def get_sglang_server_urls(self) -> list[str]:
        """Get base URLs of all SGLang servers.

        Returns:
            List of base URLs (e.g., ["http://localhost:30000", "http://localhost:30001"])
        """
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        # Get base URLs from all workers (only primary workers, TP rank 0)
        # Use run_rank_0_only_axes to only get URLs from primary workers
        futures = self.worker_group.run_all_workers_single_data(
            "get_base_url",
            run_rank_0_only_axes=["tensor_parallel"],
        )
        urls = ray.get(futures)
        # Filter out None values and return unique URLs
        return list(set(url for url in urls if url is not None))

    def get_sglang_url_to_gpu_uuids(self) -> dict[str, list[str]]:
        """Get mapping from SGLang server URL to list of GPU UUIDs it uses.

        Returns:
            Dict mapping server URL to list of GPU UUIDs
            e.g., {"http://localhost:30000": ["GPU-aaa", "GPU-bbb"], ...}
        """
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        # Get base URLs and GPU UUIDs from all primary workers (TP rank 0)
        futures_url = self.worker_group.run_all_workers_single_data(
            "get_base_url",
            run_rank_0_only_axes=["tensor_parallel"],
        )
        futures_uuids = self.worker_group.run_all_workers_single_data(
            "get_gpu_uuids",
            run_rank_0_only_axes=["tensor_parallel"],
        )

        urls = ray.get(futures_url)
        uuids_list = ray.get(futures_uuids)

        # Create mapping
        url_to_uuids = {}
        for url, uuids in zip(urls, uuids_list):
            if url is not None and uuids is not None:
                url_to_uuids[url] = uuids

        return url_to_uuids

    def _memory_saver_enabled(self) -> bool:
        """Whether the colocated server time-shares GPU via memory occupation.

        Gated by sglang_cfg.enable_memory_saver: when set, the server is launched
        with SGLang's memory-saver adapter and exposes
        release/resume_memory_occupation, so prepare_for_generation/
        finish_generation drive the wake/sleep lifecycle. When unset the server is
        always-on and both hooks are no-ops (the historical behavior).
        """
        return bool(self.sglang_cfg.get("enable_memory_saver", False))

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Reacquire the colocated server's GPU memory before a generation round.

        SGLang is always colocated in dockyard (it shares GPUs with the trainer),
        so with memory-saver enabled the server releases its GPU memory during the
        training phase (see finish_generation) and must resume it here. An optional
        ``tags`` kwarg (e.g. ["weights"], ["kv_cache"]) selects a partial resume:
        the HTTP weight-sync lifecycle calls this twice — once to stage refreshed
        weights, once to rebuild the KV cache. With memory-saver off the server is
        always-on and this is a no-op.
        """
        if not self._memory_saver_enabled():
            return True
        futures = self.worker_group.run_all_workers_single_data(
            "wake_up",
            run_rank_0_only_axes=["tensor_parallel"],
            tags=kwargs.get("tags"),
        )
        try:
            results = ray.get(futures, timeout=300)
        except Exception as exc:
            logger.error(f"[sglang colocation] resume_memory_occupation failed: {exc}")
            return False
        return all(r is not False for r in results)

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Release the colocated server's GPU memory at the end of a round.

        With memory-saver enabled the server releases all GPU memory occupation so
        the co-resident trainer can use it during the training phase;
        prepare_for_generation resumes it next round. With memory-saver off this is
        a no-op: the prefix/KV cache is deliberately preserved between rollout
        batches so SGLang's RadixAttention prefix caching stays warm (it is flushed
        only at weight-sync time via invalidate_kv_cache()).
        """
        if not self._memory_saver_enabled():
            return True
        futures = self.worker_group.run_all_workers_single_data(
            "sleep",
            run_rank_0_only_axes=["tensor_parallel"],
            tags=kwargs.get("tags"),
        )
        try:
            results = ray.get(futures, timeout=300)
        except Exception as exc:
            logger.error(f"[sglang colocation] release_memory_occupation failed: {exc}")
            return False
        return all(r is not False for r in results)

    def shutdown(self) -> bool:
        """Shut down all SGLang workers and clean up resources."""
        try:
            # Use the worker group's shutdown method with the worker's cleanup method
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            logger.error(f"Error during SGLang policy shutdown: {e}")
            return False

    def __del__(self) -> None:
        """Shuts down the worker groups when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call shutdown() and the pointer to
        the object is lost due to leaving a function scope. It's always recommended that the
        user calls shutdown().
        """
        self.shutdown()

    def invalidate_kv_cache(self) -> bool:
        """Invalidate KV cache before weight updates.

        This flushes the cache before weight updates to clear stale cache.
        Only primary workers (TP rank 0, model owners) will flush their cache.

        Returns:
            bool: True if all caches were flushed successfully, False otherwise
        """
        try:
            futures = self.worker_group.run_all_workers_single_data(
                "invalidate_kv_cache",
                run_rank_0_only_axes=["tensor_parallel"],
            )
            results = ray.get(futures)
            results = [r for r in results if r is not None]
            success = all(result for result in results) if results else True
            if success:
                logger.info(
                    "[sglang refit] All SGLang server caches flushed successfully"
                )
            else:
                logger.warning(
                    "[sglang refit] WARNING - Some SGLang server caches failed to flush"
                )
            return success
        except Exception as e:
            logger.error(f"[sglang refit] Error flushing SGLang caches: {e}")
            return False
