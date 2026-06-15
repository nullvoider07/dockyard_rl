"""Distributed Ray worker groups."""

import importlib
import logging
import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, cast
import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tqdm import tqdm

from dockyard_rl.distributed.named_sharding import NamedSharding
from dockyard_rl.distributed.virtual_cluster import RayVirtualCluster
from dockyard_rl.distributed.worker_groups_utils import (
    get_nsight_config_if_pattern_matches,
    recursive_merge_options,
)

logger = logging.getLogger(__name__)

# MultiWorkerFuture
@dataclass
class MultiWorkerFuture:
    """Container for Ray futures with associated worker routing metadata."""

    futures: list[ray.ObjectRef]
    return_from_workers: Optional[list[int]] = None
    called_workers: Optional[list[int]] = None

    def get_results(
        self,
        worker_group: "RayWorkerGroup",
        return_generators_as_proxies: bool = False,
    ) -> list[Any]:
        """Resolve futures, optionally filtering by return_from_workers.

        Args:
            worker_group:
                The RayWorkerGroup that spawned the futures; its
                worker_to_tied_group_index mapping drives deduplication.
            return_generators_as_proxies:
                When True, ObjectRefGenerators are returned as-is instead of
                being consumed.  The caller is then responsible for iteration.

        Returns:
            List of resolved results.
        """
        from ray import ObjectRef, ObjectRefGenerator

        if return_generators_as_proxies:
            if self.return_from_workers is None:
                return self.futures

            if self.called_workers is not None:
                called_map = {
                    g: i for i, g in enumerate(self.called_workers)
                }
                return [
                    self.futures[called_map[w]]
                    for w in self.return_from_workers
                    if w in called_map and called_map[w] < len(self.futures)
                ]
            return [
                self.futures[i]
                for i in self.return_from_workers
                if i < len(self.futures)
            ]

        # Expand generators into individual ObjectRefs before ray.get().
        object_refs: list[ObjectRef] = []
        has_generator = False
        for fut in self.futures:
            if isinstance(fut, ObjectRefGenerator):
                for ref in cast(Any, fut):
                    object_refs.append(ref)
                has_generator = True
            else:
                object_refs.append(fut)

        all_results = ray.get(object_refs)

        # Streaming mode — return all chunks in order.
        if has_generator:
            return all_results

        if self.return_from_workers is not None:
            if self.called_workers is not None:
                worker_to_idx = {
                    w: i for i, w in enumerate(self.called_workers)
                }
                valid = [
                    w for w in self.return_from_workers if w in worker_to_idx
                ]
                return [all_results[worker_to_idx[w]] for w in valid]
            return [all_results[w] for w in self.return_from_workers]

        return all_results

# RayWorkerBuilder
class RayWorkerBuilder:
    """Factory for Ray actors placed inside a specific placement group bundle.

    The builder is callable (synchronous) and also exposes
    create_worker_async() for batched non-blocking startup.

    Worker options precedence (lowest → highest):
        1. Worker class ``_default_options`` attribute.
        2. Options returned by ``worker_class.configure_worker()``.
        3. Scheduling strategy set by this builder (non-overridable).
    """

    @ray.remote
    class IsolatedWorkerInitializer:
        """Trampoline actor that constructs the real worker in isolation.

        Ray does not support passing actor handles as constructor arguments.
        The initializer sidesteps this by being constructed first, then
        spawning the target actor from within a Ray worker context.

        Keeping a reference to the initializer prevents Ray from GC-ing the
        child actor when the handle returned from create_worker() goes out of
        scope in the driver.
        """

        def __init__(
            self,
            ray_actor_class_fqn: str,
            *init_args: Any,
            **init_kwargs: Any,
        ) -> None:
            self.ray_actor_class_fqn = ray_actor_class_fqn
            self.init_args   = init_args
            self.init_kwargs = init_kwargs

        def create_worker(
            self,
            placement_group: PlacementGroup,
            placement_group_bundle_index: int,
            num_gpus: int,
            bundle_indices: Optional[tuple] = None,
            **extra_options: Optional[dict[str, Any]],
        ):
            """Construct and return the target Ray actor.

            See RayWorkerBuilder.__call__ for option precedence docs.
            """
            module_name, class_name = self.ray_actor_class_fqn.rsplit(".", 1)
            module       = importlib.import_module(module_name)
            worker_class = getattr(module, class_name)
            worker_kwargs= dict(self.init_kwargs)

            default_options = getattr(worker_class, "_default_options", {})
            options = recursive_merge_options(default_options, extra_options)

            if hasattr(worker_class, "configure_worker"):
                resources, env_vars, init_kw = worker_class.configure_worker(
                    num_gpus=num_gpus,
                    bundle_indices=bundle_indices,
                )
                if resources and "num_gpus" in resources:
                    num_gpus = resources["num_gpus"]
                if env_vars:
                    options.setdefault("runtime_env", {}).setdefault(
                        "env_vars", {}
                    ).update(env_vars)
                if init_kw:
                    worker_kwargs.update(init_kw)

            options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=placement_group_bundle_index,
                placement_group_capture_child_tasks=True,
            )
            options["num_gpus"] = num_gpus

            return worker_class.options(**options).remote(
                *self.init_args, **worker_kwargs
            )

    def __init__(
        self,
        ray_actor_class_fqn: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.ray_actor_class_fqn = ray_actor_class_fqn
        self.args   = args
        self.kwargs = kwargs

    def create_worker_async(
        self,
        placement_group: PlacementGroup,
        placement_group_bundle_index: int,
        num_gpus: float | int,
        bundle_indices: Optional[tuple[int, list[int]]] = None,
        **extra_options: Any,
    ) -> tuple[ray.ObjectRef, ray.actor.ActorHandle]:
        """Start actor creation asynchronously.

        Returns:
            (worker_future, initializer_actor) — caller must hold the
            initializer_actor reference to prevent GC of the child actor.
        """
        options = deepcopy(extra_options)
        initializer = cast(Any, self.IsolatedWorkerInitializer).options(
            runtime_env=options.get("runtime_env", {})
        ).remote(self.ray_actor_class_fqn, *self.args, **self.kwargs)

        create_worker_method = cast(Any, initializer.create_worker)
        worker_future = create_worker_method.remote(
            placement_group,
            placement_group_bundle_index,
            num_gpus,
            bundle_indices,
            **options,
        )
        return worker_future, initializer

    def __call__(
        self,
        placement_group: PlacementGroup,
        placement_group_bundle_index: int,
        num_gpus: float | int,
        bundle_indices: Optional[tuple[int, list[int]]] = None,
        **extra_options: Any,
    ) -> ray.actor.ActorHandle:
        """Synchronously create and return a Ray actor."""
        future, initializer = self.create_worker_async(
            placement_group,
            placement_group_bundle_index,
            num_gpus,
            bundle_indices,
            **extra_options,
        )
        worker = ray.get(future)
        # Hold the initializer to prevent GC of the child actor.
        worker._RAY_INITIALIZER_ACTOR_REF_TO_AVOID_GC = initializer
        return worker

# RayWorkerGroup
class RayWorkerGroup:
    """Manages a group of distributed Ray actor processes.

    Handles:
    - Worker creation and placement on specific GPU (or CPU) resources.
    - Distribution of distributed training env vars
      (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, NODE_RANK).
    - Parallel method dispatch with single-data, per-worker-data, or
      sharding-aware routing.
    - Tied worker groups: multiple workers that collectively own one DP shard
      (used for tensor-parallel groups where each rank processes the same
      data but a different parameter shard).
    """

    def __init__(
        self,
        cluster: RayVirtualCluster,
        remote_worker_builder: RayWorkerBuilder,
        workers_per_node: Optional[Union[int, list[int]]] = None,
        name_prefix: str = "",
        bundle_indices_list: Optional[list[tuple[int, list[int]]]] = None,
        sharding_annotations: Optional[NamedSharding] = None,
        env_vars: dict[str, str] = {},
    ) -> None:
        """Initialise a worker group.

        Args:
            cluster:
                RayVirtualCluster that owns the placement groups.
            remote_worker_builder:
                Builder that spawns individual Ray actors.
            workers_per_node:
                Workers to launch per node.  None → one worker per bundle.
                int → same count on every node.  list → per-node counts.
            name_prefix:
                Prefix for actor names (aids Ray dashboard debugging).
            bundle_indices_list:
                Explicit list of (pg_idx, [local_bundle_indices]) tuples.
                Each tuple defines a tied worker group on one node.
                When provided, workers_per_node is ignored.
            sharding_annotations:
                NamedSharding describing TP/PP/DP/EP axis assignments.
                Required for run_all_workers_sharded_data().
            env_vars:
                Additional environment variables forwarded to every worker.
                The current process env is always forwarded; env_vars
                provides overrides and additions.
        """
        self._workers: list[ray.actor.ActorHandle] = []
        self._worker_metadata: list[dict[str, Any]] = []
        self.cluster                = cluster
        self.name_prefix            = name_prefix
        self.sharding_annotations   = sharding_annotations
        self.dp_leader_worker_indices: list[int] = []

        if bundle_indices_list is None:
            bundle_indices_list = self._build_bundle_indices(workers_per_node)

        self._create_workers_from_bundle_indices(
            remote_worker_builder,
            bundle_indices_list,
            env_vars=env_vars,
        )

    # Properties
    @property
    def workers(self) -> list[ray.actor.ActorHandle]:
        return self._workers

    @property
    def worker_metadata(self) -> list[dict[str, Any]]:
        return self._worker_metadata

    @property
    def dp_size(self) -> int:
        """Number of data-parallel shards."""
        return len(self.dp_leader_worker_indices)

    # DP leader lookup
    def get_dp_leader_worker_idx(self, dp_shard_idx: int) -> int:
        """Return the global worker index of the DP leader for a given shard.

        Args:
            dp_shard_idx: Zero-based DP shard index.

        Raises:
            IndexError: If dp_shard_idx is out of range.
        """
        if not (0 <= dp_shard_idx < len(self.dp_leader_worker_indices)):
            raise IndexError(
                f"DP shard index {dp_shard_idx} out of range "
                f"[0, {len(self.dp_leader_worker_indices) - 1}]."
            )
        return self.dp_leader_worker_indices[dp_shard_idx]

    # Single-worker dispatch
    def run_single_worker_single_data(
        self,
        method_name: str,
        worker_idx: int,
        **kwargs: Any,
    ) -> ray.ObjectRef:
        """Call a method on one specific worker.

        Args:
            method_name: Remote method to call.
            worker_idx:  Index into self.workers.
            **kwargs:    Keyword arguments (no positional args allowed; see
                         upstream issue #582).

        Returns:
            Ray ObjectRef for the result.
        """
        worker = self.workers[worker_idx]
        try:
            method = cast(Any, getattr(worker, method_name))
        except AttributeError as exc:
            logger.error(
                "Supported methods: %s",
                list(worker._method_shells.keys()),
            )
            raise exc
        return method.remote(**kwargs)

    # Broadcast dispatch
    def run_all_workers_single_data(
        self,
        method_name: str,
        run_rank_0_only_axes: list[str] | None = None,
        **kwargs: Any,
    ) -> list[ray.ObjectRef]:
        """Call a method on all (or a filtered subset of) workers with the
        same arguments.

        Uses ray.put() to serialise kwargs once, then shares the same object
        refs across all workers — avoids redundant serialisation for large
        tensors replicated across many ranks.

        Args:
            method_name:          Remote method to invoke.
            run_rank_0_only_axes: Axes along which only rank-0 workers run.
                                  Workers at rank > 0 on any of these axes
                                  are skipped.
            **kwargs:             Arguments forwarded to every worker.

        Returns:
            List of Ray ObjectRefs.
        """
        if run_rank_0_only_axes is None:
            run_rank_0_only_axes = []

        # Serialise once via ray.put; all workers share the same object refs.
        put_kwargs = {k: ray.put(v) for k, v in kwargs.items()}

        futures = []
        for worker_idx, worker in enumerate(self.workers):
            if not self._should_run(worker_idx, run_rank_0_only_axes):
                continue
            try:
                method = cast(Any, getattr(worker, method_name))
            except AttributeError as exc:
                logger.error(
                    "Supported methods: %s",
                    list(worker._method_shells.keys()),
                )
                raise exc
            futures.append(method.remote(**put_kwargs))

        return futures

    # Per-worker-data dispatch
    def run_all_workers_multiple_data(
        self,
        method_name: str,
        run_rank_0_only_axes: list[str] | None = None,
        common_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> list[ray.ObjectRef]:
        """Call a method on workers with per-worker argument slices.

        Each value in **kwargs must be a list whose length equals the number
        of workers that will actually run (after axis filtering).

        Args:
            method_name:          Remote method to invoke.
            run_rank_0_only_axes: Axes along which only rank-0 workers run.
            common_kwargs:        Arguments sent identically to every worker.
            **kwargs:             Per-worker argument lists.

        Returns:
            List of Ray ObjectRefs.
        """
        if run_rank_0_only_axes is None:
            run_rank_0_only_axes = []
        if common_kwargs is None:
            common_kwargs = {}

        assert len(kwargs) > 0, (
            "At least one kwarg is required in run_all_workers_multiple_data; "
            "use run_all_workers_single_data for broadcast calls."
        )

        # Validate uniform length across all kwargs.
        lengths = [len(v) for v in kwargs.values()]
        assert all(n == lengths[0] for n in lengths), (
            "All kwargs must have the same length in "
            "run_all_workers_multiple_data."
        )
        data_count = lengths[0]

        futures = []
        data_idx = 0
        for worker_idx, worker in enumerate(self.workers):
            if not self._should_run(worker_idx, run_rank_0_only_axes):
                continue
            try:
                method = getattr(worker, method_name)
            except AttributeError as exc:
                logger.error(
                    "Supported methods: %s",
                    list(worker._method_shells.keys()),
                )
                raise exc
            worker_kwargs = {k: v[data_idx] for k, v in kwargs.items()}
            futures.append(
                method.remote(**worker_kwargs, **common_kwargs)
            )
            data_idx += 1

        assert data_idx == data_count, (
            f"Dispatched {data_idx} slices but expected {data_count}."
        )
        return futures

    # Sharding-aware dispatch
    def run_all_workers_sharded_data(
        self,
        method_name: str,
        in_sharded_axes: list[str] | None = None,
        replicate_on_axes: list[str] | None = None,
        output_is_replicated: list[str] | None = None,
        make_dummy_calls_to_free_axes: bool = False,
        common_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> MultiWorkerFuture:
        """Call a method routing data slices according to the sharding layout.

        Axis semantics
        --------------
        in_sharded_axes    : Data is pre-split; send the appropriate slice.
        replicate_on_axes  : Data is replicated to all workers on these axes.
        Free axes           : Only rank-0 workers receive data.

        output_is_replicated: Axes along which the output is the same across
                              workers; only rank-0 results are returned.

        Args:
            method_name:
                Remote method to invoke.
            in_sharded_axes:
                Axes along which data is sharded.
            replicate_on_axes:
                Axes along which data is replicated.
            output_is_replicated:
                Axes whose workers produce identical outputs (return from
                rank 0 only).
            make_dummy_calls_to_free_axes:
                If True, call the method with None arguments on workers that
                would otherwise be skipped.  Needed when workers must
                collectively synchronise (e.g. collective NCCL ops) even if
                they don't hold data.
            common_kwargs:
                Arguments broadcast to all called workers.
            **kwargs:
                Per-axis-slice argument lists.

        Returns:
            MultiWorkerFuture wrapping futures and routing metadata.
        """
        if self.sharding_annotations is None:
            raise ValueError(
                "sharding_annotations must be set to use sharded data dispatch."
            )
        if in_sharded_axes is None:
            in_sharded_axes = []
        if replicate_on_axes is None:
            replicate_on_axes = []
        if output_is_replicated is None:
            output_is_replicated = []
        if common_kwargs is None:
            common_kwargs = {}

        # Validate axes.
        for ax in in_sharded_axes + replicate_on_axes:
            if ax not in self.sharding_annotations.names:
                raise ValueError(
                    f"Axis {ax!r} not in sharding annotations "
                    f"{self.sharding_annotations.names}."
                )
        overlap = set(in_sharded_axes) & set(replicate_on_axes)
        if overlap:
            raise ValueError(
                f"Axes cannot be both sharded and replicated: {overlap}"
            )

        # Use ray.put on replicated kwargs to avoid re-serialisation.
        replicate_degree = math.prod(
            self.sharding_annotations.get_axis_size(ax)
            for ax in replicate_on_axes
        )
        if replicate_degree > 1:
            def _put_nested(val: Any, depth: int) -> Any:
                if depth == 0:
                    return ray.put(val)
                return [_put_nested(v, depth - 1) for v in val]

            kwargs = {
                k: _put_nested(v, len(in_sharded_axes))
                for k, v in kwargs.items()
            }

        futures: list[ray.ObjectRef] = []
        called_workers: list[int] = []
        return_from_workers: list[int] = []

        for worker_idx, worker in enumerate(self._workers):
            coords = self.sharding_annotations.get_worker_coords(worker_idx)

            should_receive = True
            should_return  = True

            for axis in self.sharding_annotations.names:
                if axis not in coords:
                    continue
                # Free axis (not in sharded or replicated): only rank 0 runs.
                if (
                    axis not in in_sharded_axes
                    and axis not in replicate_on_axes
                    and coords[axis] != 0
                ):
                    should_receive = False
                    should_return  = False
                    break
                if axis in output_is_replicated and coords[axis] != 0:
                    should_return = False

            if should_return:
                return_from_workers.append(worker_idx)

            if should_receive:
                worker_kwargs = dict(kwargs)
                for axis in in_sharded_axes:
                    if axis in coords:
                        idx = coords[axis]
                        worker_kwargs = {
                            k: v[idx] for k, v in worker_kwargs.items()
                        }
                method = cast(Any, getattr(worker, method_name))
                future = method.remote(**worker_kwargs, **common_kwargs)
                futures.append(future)
                called_workers.append(worker_idx)
            elif make_dummy_calls_to_free_axes:
                dummy_kwargs = {k: None for k in kwargs}
                method = cast(Any, getattr(worker, method_name))
                future = method.remote(**dummy_kwargs, **common_kwargs)
                futures.append(future)
                called_workers.append(worker_idx)

        return MultiWorkerFuture(
            futures=futures,
            called_workers=called_workers,
            return_from_workers=return_from_workers,
        )

    def get_all_worker_results(
        self,
        future_bundle: MultiWorkerFuture,
        return_generators_as_proxies: bool = False,
    ) -> list[Any]:
        """Resolve a MultiWorkerFuture, deduplicating tied-group results."""
        return future_bundle.get_results(
            self,
            return_generators_as_proxies=return_generators_as_proxies,
        )

    # Shutdown
    def shutdown(
        self,
        cleanup_method: Optional[str] = None,
        timeout: Optional[float] = 30.0,
        force: bool = False,
    ) -> bool:
        """Shut down all workers.

        Args:
            cleanup_method: Optional remote method to call before kill.
            timeout:        Seconds to wait for graceful cleanup.
            force:          Skip graceful cleanup even if cleanup_method set.

        Returns:
            True if all workers were cleanly terminated.
        """
        if not self._workers:
            return True

        success = True

        if cleanup_method is not None and not force:
            try:
                futures = self.run_all_workers_single_data(cleanup_method)
                if timeout is not None:
                    ray.get(futures, timeout=timeout)
                else:
                    ray.get(futures)
            except (
                ray.exceptions.RayTaskError,
                ray.exceptions.GetTimeoutError,
            ) as exc:
                success = False
                logger.warning(
                    "Graceful shutdown failed: %s — forcing kill.", exc
                )

        initializers_to_kill = []
        for worker in self._workers:
            if hasattr(worker, "_RAY_INITIALIZER_ACTOR_REF_TO_AVOID_GC"):
                ref = getattr(
                    worker, "_RAY_INITIALIZER_ACTOR_REF_TO_AVOID_GC", None
                )
                if ref is not None:
                    initializers_to_kill.append(ref)
            try:
                ray.kill(worker)
            except Exception as exc:
                success = False
                logger.warning("Error killing worker: %s", exc)

        for init in initializers_to_kill:
            try:
                ray.kill(init)
            except Exception as exc:
                logger.warning("Error killing initializer actor: %s", exc)

        self._workers = []
        self._worker_metadata = []
        return success

    # Internal helpers
    def _should_run(
        self,
        worker_idx: int,
        run_rank_0_only_axes: list[str],
    ) -> bool:
        """Return True if worker should participate in a broadcast call."""
        if not run_rank_0_only_axes or self.sharding_annotations is None:
            return True
        coords = self.sharding_annotations.get_worker_coords(worker_idx)
        for axis in self.sharding_annotations.names:
            if axis in run_rank_0_only_axes and coords.get(axis, 0) != 0:
                return False
        return True

    def _build_bundle_indices(
        self,
        workers_per_node: Optional[Union[int, list[int]]],
    ) -> list[tuple[int, list[int]]]:
        """Construct bundle_indices_list from workers_per_node."""
        pgs = self.cluster.get_placement_groups()

        if len(pgs) == 1:
            workers_per_pg = [pgs[0].bundle_count]
        else:
            workers_per_pg = [pg.bundle_count for pg in pgs]

        if workers_per_node is None:
            workers_per_pg = [pg.bundle_count for pg in pgs]
        elif isinstance(workers_per_node, int):
            workers_per_pg = [workers_per_node] * len(pgs)
        elif isinstance(workers_per_node, list):
            if len(workers_per_node) == 1 and len(pgs) == 1:
                workers_per_pg = workers_per_node
            elif len(workers_per_node) != len(pgs):
                raise ValueError(
                    f"workers_per_node list length ({len(workers_per_node)}) "
                    f"must match placement group count ({len(pgs)})."
                )
            else:
                workers_per_pg = workers_per_node
        else:
            raise ValueError(
                "workers_per_node must be None, int, or list."
            )

        for i, (pg, wc) in enumerate(zip(pgs, workers_per_pg)):
            if wc > pg.bundle_count:
                raise ValueError(
                    f"Placement group {i} has {pg.bundle_count} bundles "
                    f"but {wc} workers were requested."
                )

        indices: list[tuple[int, list[int]]] = []
        for pg_idx, wc in enumerate(workers_per_pg):
            for bundle_idx in range(wc):
                indices.append((pg_idx, [bundle_idx]))
        return indices

    def _create_workers_from_bundle_indices(
        self,
        remote_worker_builder: RayWorkerBuilder,
        bundle_indices_list: list[tuple[int, list[int]]],
        env_vars: dict[str, str] = {},
    ) -> None:
        """Spawn all workers asynchronously then collect them.

        Distributes standard distributed training env vars (RANK,
        LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, NODE_RANK) and
        optionally injects Nsight profiling config for matching actors.
        """
        master_addr, master_port = self.cluster.get_master_address_and_port()
        self.master_address = master_addr
        self.master_port    = master_port

        # Merge caller-supplied overrides on top of the current process env.
        base_env: dict[str, str] = {
            k: v for k, v in os.environ.items() if k not in env_vars
        }
        base_env.update(env_vars)

        # Discover IP+port for every bundle upfront (parallel ray.get).
        pgs = self.cluster.get_placement_groups()
        addr_list: list[str] = []
        port_list: list[int] = []
        for pg_idx, local_bundle_indices in bundle_indices_list:
            for bundle_idx in local_bundle_indices:
                addr, port = self.cluster.get_available_address_and_port(
                    pg_idx, bundle_idx
                )
                addr_list.append(addr)
                port_list.append(port)

        self.world_size = sum(
            len(indices) for _, indices in bundle_indices_list
        )

        # Async worker creation
        worker_futures: list[tuple[ray.ObjectRef, ray.actor.ActorHandle]] = []
        worker_info:    list[dict[str, Any]] = []
        global_rank = 0

        for group_idx, (pg_idx, local_bundle_indices) in enumerate(
            bundle_indices_list
        ):
            pg = pgs[0] if len(pgs) == 1 else pgs[pg_idx]
            is_parallel = len(local_bundle_indices) > 1

            for local_rank, bundle_idx in enumerate(local_bundle_indices):
                worker_env = deepcopy(base_env)

                # Drop Ray internal vars that workers must set themselves.
                for key in (
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
                    "RAY_CLIENT_MODE",
                    "RAY_JOB_ID",
                    "RAY_LD_PRELOAD",
                    "RAY_RAYLET_PID",
                    "RAY_USAGE_STATS_ENABLED",
                ):
                    worker_env.pop(key, None)

                worker_env.update({
                    "RANK":                str(global_rank),
                    "LOCAL_RANK":          str(bundle_idx),
                    "WORLD_SIZE":          str(self.world_size),
                    "MASTER_ADDR":         master_addr,
                    "MASTER_PORT":         str(master_port),
                    "NODE_RANK":           str(pg_idx),
                    "AVAILABLE_ADDR_LIST": str(addr_list),
                    "AVAILABLE_PORT_LIST": str(port_list),
                })

                # Only the first worker in a tied group owns bundle_indices
                # (identifies itself as the model-parameter owner).
                worker_bundle_indices: Optional[tuple[int, list[int]]] = None
                if local_rank == 0:
                    worker_bundle_indices = (pg_idx, local_bundle_indices)
                    self.dp_leader_worker_indices.append(global_rank)

                actor_name = (
                    f"{self.name_prefix}-grp{group_idx}-{local_rank}"
                    if is_parallel
                    else f"{self.name_prefix}-{pg_idx}-{bundle_idx}"
                )

                num_gpus = (
                    1 / self.cluster.max_colocated_worker_groups
                    if self.cluster.use_gpus
                    else 0
                )

                runtime_env: dict[str, Any] = {
                    "env_vars": worker_env,
                    # ubuntu-swe: all deps are in the system Python.
                    # py_executable is always sys.executable — no uv venvs.
                    "py_executable": sys.executable,
                }

                # Optionally inject Nsight profiling config.
                nsight_cfg = get_nsight_config_if_pattern_matches(actor_name)
                if nsight_cfg:
                    runtime_env.update(nsight_cfg)

                future, initializer = remote_worker_builder.create_worker_async(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx,
                    num_gpus=num_gpus,
                    bundle_indices=worker_bundle_indices,
                    runtime_env=runtime_env,
                    name=actor_name,
                )

                worker_futures.append((future, initializer))
                worker_info.append({
                    "group_idx":      group_idx,
                    "node_idx":       pg_idx,
                    "local_rank":     local_rank,
                    "global_rank":    global_rank,
                    "name":           actor_name,
                    "bundle_indices": worker_bundle_indices,
                    "dp_shard_idx":   group_idx,
                })

                global_rank += 1

        # Collect workers with tqdm progress bar
        refs        = [f for f, _ in worker_futures]
        remaining   = list(refs)
        start       = time.perf_counter()
        n           = len(refs)

        with tqdm(
            total=n,
            desc=f"Initialising {self.name_prefix} workers",
            unit="worker",
        ) as pbar:
            while remaining:
                ready, remaining = ray.wait(
                    remaining, num_returns=1, timeout=None
                )
                pbar.update(len(ready))

        workers = ray.get(refs)
        elapsed = time.perf_counter() - start
        logger.info(
            "✓ %d %s workers initialised in %.2fs",
            n,
            self.name_prefix,
            elapsed,
        )

        for idx, (worker, (_, initializer)) in enumerate(
            zip(workers, worker_futures)
        ):
            worker._RAY_INITIALIZER_ACTOR_REF_TO_AVOID_GC = initializer
            self._workers.append(worker)
            info = worker_info[idx]
            self._worker_metadata.append({
                "node_idx":       info["node_idx"],
                "local_rank":     info["local_rank"],
                "global_rank":    info["global_rank"],
                "name":           info["name"],
                "bundle_indices": info["bundle_indices"],
                "dp_shard_idx":   info["group_idx"],
            })