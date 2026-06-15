"""Ray virtual cluster management for Project Dockyard.

Provides RayVirtualCluster: a thin wrapper around Ray placement groups
that presents a logical compute cluster (trainer fleet, inference fleet,
or sandbox fleet) as a typed object with stable address/port discovery.
"""

import logging
import os
import socket
import time
from typing import NotRequired, Optional, TypedDict

import ray
from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logger = logging.getLogger(__name__)

class ClusterConfig(TypedDict):
    gpus_per_node: int
    num_nodes: int
    # Master / TCPStore port range. Kept below the OS ephemeral range
    # (32768-60999 on stock Linux) to avoid TOCTOU collisions with
    # kernel-assigned source ports. See ray.sub for the full port layout.
    master_port_range_low: NotRequired[int]
    master_port_range_high: NotRequired[int]

# IP / port helpers
#
# Master / TCPStore ports bind within an explicit range kept below the OS
# ephemeral range (32768-60999 on stock Linux) to avoid TOCTOU collisions
# with kernel-assigned source ports. The range defaults to the
# DOCKYARD_MASTER_PORT_RANGE_LOW/HIGH env vars, then to the constants below.
DEFAULT_MASTER_PORT_RANGE_LOW = 25000
DEFAULT_MASTER_PORT_RANGE_HIGH = 28000


def _master_port_range() -> tuple[int, int]:
    low = int(os.environ.get(
        "DOCKYARD_MASTER_PORT_RANGE_LOW", DEFAULT_MASTER_PORT_RANGE_LOW))
    high = int(os.environ.get(
        "DOCKYARD_MASTER_PORT_RANGE_HIGH", DEFAULT_MASTER_PORT_RANGE_HIGH))
    return low, high


@ray.remote  # pragma: no cover
def _get_node_ip_and_free_port(
    port_range_low: Optional[int] = None,
    port_range_high: Optional[int] = None,
) -> tuple[str, int]:
    return _get_node_ip_local(), _get_free_port_local(
        port_range_low, port_range_high)


def _get_node_ip_local() -> str:
    return ray._private.services.get_node_ip_address()  # type: ignore[attr-defined]


def _bind_socket_in_range(
    sock: socket.socket,
    port_range_low: int,
    port_range_high: int,
    max_retries: int = 50,
) -> int:
    """Bind sock to a random free port in [port_range_low, port_range_high).

    Raises RuntimeError after max_retries failed attempts.
    """
    import random

    for _ in range(max_retries):
        port = random.randint(port_range_low, port_range_high - 1)
        try:
            sock.bind(("", port))
            return port
        except OSError:
            continue
    raise RuntimeError(
        f"Could not find a free port in range "
        f"[{port_range_low}, {port_range_high}) after {max_retries} attempts."
    )


def _get_free_port_local(
    port_range_low: Optional[int] = None,
    port_range_high: Optional[int] = None,
) -> int:
    default_low, default_high = _master_port_range()
    low = default_low if port_range_low is None else port_range_low
    high = default_high if port_range_high is None else port_range_high
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        port = _bind_socket_in_range(s, low, high)
        s.listen(1)
    return port

# Diagnostic actor
@ray.remote(num_gpus=1)
class GetGPUIdActor:  # pragma: no cover
    """Utility actor that returns the GPU id assigned to the current worker."""

    def get_gpu_id(self):
        return ray.get_gpu_ids()[0]

# Exceptions
class ResourceInsufficiencyError(Exception):
    """Raised when the cluster cannot satisfy a requested placement config."""

# RayVirtualCluster
class RayVirtualCluster:
    """A logical distributed cluster backed by Ray placement groups.

    Concepts
    --------
    Bundle   — one resource-allocation unit (1 GPU + N CPUs).
    Node     — one entry in bundle_ct_per_node_list; maps to one
               per-node placement group or to a slice of a unified group.
    Worker   — a Ray actor scheduled into a specific bundle.

    The cluster can be created in two topologies:

    Per-node groups (default)
        One STRICT_PACK placement group per node entry.  Workers on the
        same node share NVLink bandwidth.  Used for trainers and per-node
        inference replicas.

    Unified group (use_unified_pg=True)
        A single placement group spanning all bundles across all nodes.
        Required when TP > gpus_per_node (cross-node tensor parallelism).
        In this mode, _get_sorted_bundle_indices() reorders bundles by
        (node_id, gpu_id) so RANK=0 is always on the first physical GPU.
    """

    def __init__(
        self,
        bundle_ct_per_node_list: list[int],
        use_gpus: bool = True,
        max_colocated_worker_groups: int = 1,
        num_gpus_per_node: int = 8,
        name: str = "",
        placement_group_strategy: str = "SPREAD",
    ) -> None:
        """Initialise a virtual cluster.

        Args:
            bundle_ct_per_node_list:
                Number of GPU bundles per logical node.
                e.g. [8, 8] creates two nodes with 8 bundles each.
            use_gpus:
                Whether to claim GPU resources in bundles.
            max_colocated_worker_groups:
                Maximum number of worker groups colocated on this cluster.
                Controls how many CPUs are reserved per bundle
                (= max_colocated_worker_groups CPUs/bundle).
            num_gpus_per_node:
                GPUs per node — used only for assertion if use_gpus=True.
            name:
                Prefix for placement group names (aids debugging).
            placement_group_strategy:
                Ray placement group strategy for the top-level group
                (SPREAD | PACK | STRICT_PACK).  Per-node sub-groups
                always use STRICT_PACK regardless of this setting.
        """
        if use_gpus:
            assert num_gpus_per_node > 0, (
                "num_gpus_per_node must be > 0 when use_gpus=True"
            )

        self._bundle_ct_per_node_list    = bundle_ct_per_node_list
        self._world_size                 = sum(bundle_ct_per_node_list)
        self._node_placement_groups: Optional[list[PlacementGroup]] = None
        self._sorted_bundle_indices: Optional[list[int]] = None

        self.num_gpus_per_node             = num_gpus_per_node
        self.use_gpus                      = use_gpus
        self.max_colocated_worker_groups   = max_colocated_worker_groups
        self.name                          = name
        self.placement_group_strategy      = placement_group_strategy

    # Public placement group API
    def _init_placement_groups(
        self,
        strategy: str | None = None,
        use_unified_pg: bool = False,
    ) -> list[PlacementGroup]:
        """Create placement groups, with exponential-backoff retry.

        Idempotent: returns cached groups on repeat calls.

        Args:
            strategy:        Override for placement_group_strategy.
            use_unified_pg:  Create a single unified group (cross-node TP).

        Returns:
            List of created PlacementGroup objects.
        """
        if self._node_placement_groups is not None:
            return self._node_placement_groups

        if strategy is None:
            strategy = self.placement_group_strategy

        max_retries = int(
            os.environ.get("DOCKYARD_VIRTUAL_CLUSTER_MAX_RETRIES", 6)
        )
        assert max_retries > 0, (
            f"DOCKYARD_VIRTUAL_CLUSTER_MAX_RETRIES={max_retries} "
            "must be a positive integer."
        )

        for attempt in range(max_retries):
            try:
                self._node_placement_groups = (
                    self._create_placement_groups_internal(
                        strategy, use_unified_pg
                    )
                )
                if use_unified_pg and self.use_gpus:
                    self._sorted_bundle_indices = (
                        self._get_sorted_bundle_indices()
                    )
                return self._node_placement_groups
            except ResourceInsufficiencyError as exc:
                delay = 2 ** attempt
                logger.warning(
                    "%s — retry %d/%d in %ds.",
                    exc, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)

        raise ResourceInsufficiencyError(
            f"Placement group creation failed after {max_retries} retries. "
            "Cluster resources may be insufficient or unstable. "
            "Check cluster logs and DOCKYARD_VIRTUAL_CLUSTER_MAX_RETRIES."
        )

    def get_placement_groups(self) -> list[PlacementGroup]:
        """Return placement groups, initialising lazily if needed."""
        if self._node_placement_groups is None:
            self._init_placement_groups()
        assert self._node_placement_groups is not None
        return [
            pg for pg in self._node_placement_groups if pg.bundle_specs
        ]

    # Topology metadata
    def world_size(self) -> int:
        """Total number of bundles (= number of ranks)."""
        return self._world_size

    def node_count(self) -> int:
        """Number of logical nodes with at least one bundle."""
        return sum(1 for c in self._bundle_ct_per_node_list if c > 0)

    # Address / port discovery
    def get_available_address_and_port(
        self,
        pg_idx: int,
        bundle_idx: int,
        port_range_low: Optional[int] = None,
        port_range_high: Optional[int] = None,
    ) -> tuple[str, int]:
        """Discover the IP and a free port for the given bundle.

        Launches a zero-CPU Ray task on the target bundle to read the
        node's IP and bind a free port within the configured range.

        Args:
            pg_idx:          Index into get_placement_groups() list.
            bundle_idx:      Bundle index within the chosen placement group.
            port_range_low:  Lower bound of the port range; None falls back to
                             DOCKYARD_MASTER_PORT_RANGE_LOW / the default.
            port_range_high: Upper bound of the port range; None falls back to
                             DOCKYARD_MASTER_PORT_RANGE_HIGH / the default.

        Returns:
            (ip_address, free_port) tuple.
        """
        pgs = self.get_placement_groups()
        pg  = pgs[0] if len(pgs) == 1 else pgs[pg_idx]

        if not pg.bundle_specs:
            raise RuntimeError(
                "No valid placement groups to get address and port."
            )

        addr, port = ray.get(
            _get_node_ip_and_free_port.options(  # type: ignore[attr-defined]
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx,
                ),
                num_cpus=0,
            ).remote(port_range_low, port_range_high)  # type: ignore[call-arg]
        )
        return addr, port

    def get_master_address_and_port(
        self,
        port_range_low: Optional[int] = None,
        port_range_high: Optional[int] = None,
    ) -> tuple[str, int]:
        """Return the master (rank 0) IP and a free port.

        For unified placement groups the port is discovered on the bundle
        corresponding to the lowest physical GPU (rank 0 after sorting by
        node_id then gpu_id).  For per-node groups it is always bundle 0
        of the first placement group.

        The port is bound within [port_range_low, port_range_high), defaulting
        to the DOCKYARD_MASTER_PORT_RANGE_LOW/HIGH env vars (then the module
        constants) when either bound is None.
        """
        if not self._node_placement_groups:
            self.get_placement_groups()

        if self._sorted_bundle_indices is not None:
            return self.get_available_address_and_port(
                pg_idx=0,
                bundle_idx=self._sorted_bundle_indices[0],
                port_range_low=port_range_low,
                port_range_high=port_range_high,
            )

        return self.get_available_address_and_port(
            pg_idx=0,
            bundle_idx=0,
            port_range_low=port_range_low,
            port_range_high=port_range_high,
        )

    # Lifecycle
    def shutdown(self) -> bool:
        """Remove all placement groups and reset internal state.

        Idempotent; safe to call multiple times.

        Returns:
            True if all placement groups were removed without error.
        """
        if self._node_placement_groups is None:
            return True

        success = True
        for pg in self._node_placement_groups:
            try:
                remove_placement_group(pg)
            except Exception as exc:
                logger.warning("Error removing placement group %s: %s", pg.id, exc)
                success = False

        self._node_placement_groups = None
        self._sorted_bundle_indices = None
        return success

    def __del__(self) -> None:
        """Safety-net shutdown on GC.  Always call shutdown() explicitly."""
        self.shutdown()

    # Internal helpers
    def _create_placement_groups_internal(
        self,
        strategy: str,
        use_unified_pg: bool,
    ) -> list[PlacementGroup]:
        """Create placement groups without retry logic.

        Raises ResourceInsufficiencyError on insufficient GPU or CPU
        resources rather than letting Ray block indefinitely.
        """
        cluster_res  = ray.cluster_resources()
        avail_gpus   = int(cluster_res.get("GPU", 0))
        avail_cpus   = int(cluster_res.get("CPU", 0))

        req_gpus = sum(self._bundle_ct_per_node_list) if self.use_gpus else 0
        req_cpus = (
            sum(self._bundle_ct_per_node_list)
            * self.max_colocated_worker_groups
        )

        if self.use_gpus and req_gpus > avail_gpus:
            raise ResourceInsufficiencyError(
                f"Requested {req_gpus} GPUs but only {avail_gpus} available."
            )
        if req_cpus > avail_cpus:
            raise ResourceInsufficiencyError(
                f"Requested {req_cpus} CPUs but only {avail_cpus} available."
            )

        cpus_per_bundle = self.max_colocated_worker_groups
        gpus_per_bundle = 1 if self.use_gpus else 0

        if use_unified_pg:
            # Single group spanning all nodes — required for cross-node TP.
            all_bundles: list[dict[str, float]] = [
                {"CPU": cpus_per_bundle, "GPU": gpus_per_bundle}
                for count in self._bundle_ct_per_node_list
                for _ in range(count)
            ]
            pgs = [
                placement_group(
                    bundles=all_bundles,
                    strategy=strategy,
                    name=f"{self.name}-unified",
                )
            ]
        else:
            # One STRICT_PACK group per logical node so NVLink locality is
            # guaranteed.  The caller-supplied strategy applies at the level
            # of how these per-node groups are distributed across physical
            # nodes by the Ray scheduler, not within each group.
            pgs = []
            for node_idx, count in enumerate(self._bundle_ct_per_node_list):
                if count <= 0:
                    continue
                node_bundles: list[dict[str, float]] = [
                    {"CPU": cpus_per_bundle, "GPU": gpus_per_bundle}
                    for _ in range(count)
                ]
                pgs.append(
                    placement_group(
                        bundles=node_bundles,
                        # STRICT_PACK: all bundles of this group must be on
                        # the same physical node.  If the node is full, Ray
                        # raises immediately rather than splitting the group.
                        strategy="STRICT_PACK",
                        name=f"{self.name}-node{node_idx}",
                    )
                )

        # Wait with a generous timeout; surface a clean error on failure.
        try:
            ray.get(
                [pg.ready() for pg in pgs],
                timeout=180,
            )
        except (TimeoutError, ray.exceptions.GetTimeoutError):
            for pg in pgs:
                try:
                    remove_placement_group(pg)
                except Exception:
                    pass
            raise TimeoutError(
                "Timed out waiting for placement groups (180 s). "
                "The cluster may lack resources for the requested topology, "
                "or nodes may not yet be registered with Ray."
            )

        return pgs

    def _get_sorted_bundle_indices(self) -> Optional[list[int]]:
        """Return bundle indices sorted by (node_id, gpu_id).

        Only valid after a unified placement group has been created.
        Returns None for CPU-only or per-node cluster types.
        """
        if self._node_placement_groups is None:
            raise ValueError(
                "Placement groups must exist before calling "
                "_get_sorted_bundle_indices."
            )
        if not self.use_gpus:
            return None
        if len(self._node_placement_groups) != 1:
            return None

        pg      = self._node_placement_groups[0]
        pg_data = placement_group_table(pg)
        n       = len(pg_data["bundles"])
        node_ids= pg_data["bundles_to_node_id"]

        # Spawn lightweight actors to read each bundle's GPU id.
        info_actors = [
            GetGPUIdActor.options(
                num_cpus=0.01,
                num_gpus=0.01,
                resources=None,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
            ).remote()
            for i in range(n)
        ]
        gpu_ids = ray.get([a.get_gpu_id.remote() for a in info_actors])  # type: ignore[union-attr]
        for a in info_actors:
            ray.kill(a)  # type: ignore[arg-type]

        # Sort: primary key = node_id, secondary = gpu_id.
        bundle_infos = [(i, node_ids[i], gpu_ids[i]) for i in range(n)]
        return [
            b[0]
            for b in sorted(bundle_infos, key=lambda x: (x[1], x[2]))
        ]