"""Fleet descriptors and RayVirtualCluster builders.

Three distinct fleets operate simultaneously in the async RL loop:

  Trainer   — policy optimizer; GPU; SPREAD across nodes so each node's
               GPUs stay in their own placement group (intra-node NVLink
               for TP, inter-node RDMA for PP/DP).

  Inference — vLLM workers; GPU; PACK per node so TP ranks that share a
               node are placed together.  VllmGeneration promotes to a
               unified placement group automatically when TP exceeds
               gpus_per_node (cross-node tensor parallelism).

  Sandbox   — coding-task execution environments; CPU-only; SPREAD so
               episode slots distribute evenly across CPU nodes.  No GPU
               resources claimed, so sandbox containers never compete
               with the inference fleet for device allocation.

Usage:
    spec    = FleetSpec.from_env()
    cluster = build_cluster(spec)
    cluster._init_placement_groups()   # call before constructing workers
"""

import os
from dataclasses import dataclass

from dockyard_rl.cluster.bootstrap import (
    FLEET_INFERENCE,
    FLEET_SANDBOX,
    FLEET_TRAINER,
    get_fleet_role,
)
from dockyard_rl.distributed.virtual_cluster import (
    RayVirtualCluster,
    prepare_segment_topology,
)

# Placement strategy constants
# Pass these to FleetSpec.placement_strategy.
STRATEGY_SPREAD      = "SPREAD"
STRATEGY_PACK        = "PACK"
STRATEGY_STRICT_PACK = "STRICT_PACK"

# Per-fleet defaults.  Can be overridden via DOCKYARD_PG_STRATEGY.
_DEFAULTS: dict[str, dict] = {
    FLEET_TRAINER:   {"strategy": STRATEGY_SPREAD, "max_colocated": 1},
    FLEET_INFERENCE: {"strategy": STRATEGY_PACK,   "max_colocated": 1},
    FLEET_SANDBOX:   {"strategy": STRATEGY_SPREAD, "max_colocated": 1},
}


# Fleet specification
@dataclass(frozen=True)
class FleetSpec:
    """Immutable description of a fleet's topology and resource requirements.

    Attributes:
        role:                    "trainer" | "inference" | "sandbox"
        num_nodes:               Number of physical nodes in the fleet.
        gpus_per_node:           GPUs claimed per node (0 for sandbox).
        placement_strategy:      Ray placement group strategy string.
        max_colocated_worker_groups:
                                 Passed to RayVirtualCluster; controls
                                 CPUs allocated per bundle.
        name:                    Name prefix for placement groups.
    """
    role:                        str
    num_nodes:                   int
    gpus_per_node:               int
    placement_strategy:          str
    max_colocated_worker_groups: int = 1
    name:                        str = ""
    segment_size:                int | None = None

    def __post_init__(self) -> None:
        if self.role not in (FLEET_TRAINER, FLEET_INFERENCE, FLEET_SANDBOX):
            raise ValueError(f"Unknown fleet role: {self.role!r}")
        if self.num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {self.num_nodes}")
        if self.gpus_per_node < 0:
            raise ValueError(
                f"gpus_per_node must be >= 0, got {self.gpus_per_node}"
            )
        if self.segment_size is not None:
            if self.segment_size < 1:
                raise ValueError(
                    f"segment_size must be >= 1, got {self.segment_size}"
                )
            if self.role == FLEET_SANDBOX:
                raise ValueError(
                    "segment_size (NVLink-domain topology) is GPU-fleet only; "
                    "the CPU sandbox fleet must leave it None."
                )
            if self.num_nodes % self.segment_size != 0:
                raise ValueError(
                    f"num_nodes ({self.num_nodes}) must be divisible by "
                    f"segment_size ({self.segment_size})."
                )
        if self.role == FLEET_SANDBOX and self.gpus_per_node != 0:
            raise ValueError(
                "Sandbox fleet is CPU-only; gpus_per_node must be 0"
            )
        if self.role != FLEET_SANDBOX and self.gpus_per_node == 0:
            raise ValueError(
                f"{self.role} fleet requires gpus_per_node > 0"
            )
        # frozen=True prevents direct attribute assignment, so use
        # object.__setattr__ for the name default.
        if not self.name:
            object.__setattr__(self, "name", f"dockyard-{self.role}")

    @classmethod
    def from_env(cls) -> "FleetSpec":
        """Build a FleetSpec from environment variables.

        Required:
            DOCKYARD_FLEET_ROLE       trainer | inference | sandbox
            DOCKYARD_NUM_NODES        positive integer

        Optional (per-role defaults apply when absent):
            DOCKYARD_GPUS_PER_NODE    integer; defaults to 0 for sandbox,
                                      required for trainer/inference
            DOCKYARD_PG_STRATEGY      SPREAD | PACK | STRICT_PACK
            DOCKYARD_MAX_COLOCATED_WORKER_GROUPS   positive integer

        Raises:
            RuntimeError  if a required variable is missing.
            ValueError    if a value is out of range.
        """
        role = get_fleet_role()

        num_nodes_raw = os.environ.get("DOCKYARD_NUM_NODES", "").strip()
        if not num_nodes_raw:
            raise RuntimeError(
                "DOCKYARD_NUM_NODES is not set. "
                "Every fleet container must declare its node count."
            )
        num_nodes = int(num_nodes_raw)

        gpus_default = "0" if role == FLEET_SANDBOX else ""
        gpus_raw = os.environ.get("DOCKYARD_GPUS_PER_NODE", gpus_default).strip()
        if not gpus_raw:
            raise RuntimeError(
                "DOCKYARD_GPUS_PER_NODE is not set. "
                "trainer and inference fleets must declare GPUs per node."
            )
        gpus_per_node = int(gpus_raw)

        defaults = _DEFAULTS[role]
        
        # FIX: Ensure a string fallback so .strip().upper() is always safe
        placement_strategy = os.environ.get(
            "DOCKYARD_PG_STRATEGY", defaults["strategy"]
        )
        if placement_strategy is None:
            placement_strategy = ""
        placement_strategy = placement_strategy.strip().upper()

        # FIX: Explicitly fallback to the default string before converting to int
        max_colocated_raw = os.environ.get(
            "DOCKYARD_MAX_COLOCATED_WORKER_GROUPS"
        )
        if max_colocated_raw is not None:
            max_colocated = int(max_colocated_raw.strip())
        else:
            max_colocated = int(defaults["max_colocated"])

        # Optional NVLink-domain segment size for topology-aware placement.
        # Absent (or empty) => None => legacy unordered allocation.
        segment_raw = os.environ.get("DOCKYARD_SEGMENT_SIZE", "").strip()
        segment_size = int(segment_raw) if segment_raw else None

        return cls(
            role=role,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            placement_strategy=placement_strategy,
            max_colocated_worker_groups=max_colocated,
            segment_size=segment_size,
        )

# Virtual cluster builders
def build_cluster(spec: FleetSpec) -> RayVirtualCluster:
    """Create a RayVirtualCluster appropriate for the given FleetSpec.

    Dispatches to the per-role builder.  Call
    cluster._init_placement_groups() before constructing any workers.
    """
    builders = {
        FLEET_TRAINER:   _build_trainer_cluster,
        FLEET_INFERENCE: _build_inference_cluster,
        FLEET_SANDBOX:   _build_sandbox_cluster,
    }
    return builders[spec.role](spec)

def _segment_constraints(spec: FleetSpec) -> list[dict[str, float]] | None:
    """Per-node NVLink-domain constraints for a GPU fleet, or None.

    Returns ``None`` (legacy unordered placement) when ``segment_size`` is unset
    or when no NVLink-domain info is registered in the Ray cluster. Otherwise
    selects ``num_nodes`` segment-aligned nodes via :func:`prepare_segment_topology`
    and returns their domain-pinning dicts. The topology read is cluster-bound;
    callers only reach it with ``segment_size`` set.
    """
    if spec.segment_size is None:
        return None
    node_resource_constraints, _remaining, _topology = prepare_segment_topology(
        spec.segment_size, spec.num_nodes, role=spec.role
    )
    return node_resource_constraints

def _build_trainer_cluster(spec: FleetSpec) -> RayVirtualCluster:
    """One bundle per GPU, SPREAD across nodes.

    SPREAD ensures each node gets its own placement group so intra-node
    NVLink bandwidth is fully available for tensor-parallel communication,
    while inter-node RDMA handles pipeline and data-parallel collectives.
    With ``segment_size`` set, bundles are additionally pinned to selected
    NVLink domains and ordered by physical topology.
    """
    return RayVirtualCluster(
        bundle_ct_per_node_list=[spec.gpus_per_node] * spec.num_nodes,
        use_gpus=True,
        max_colocated_worker_groups=spec.max_colocated_worker_groups,
        num_gpus_per_node=spec.gpus_per_node,
        name=spec.name,
        placement_group_strategy=spec.placement_strategy,
        segment_size=spec.segment_size,
        node_resource_constraints=_segment_constraints(spec),
    )

def _build_inference_cluster(spec: FleetSpec) -> RayVirtualCluster:
    """One bundle per GPU, PACK per node.

    PACK keeps TP ranks on the same node together in one placement group.
    When TP > gpus_per_node (large model spanning nodes), VllmGeneration
    detects this and promotes to a unified cross-node placement group
    automatically — no change needed here. With ``segment_size`` set, bundles
    are additionally pinned to selected NVLink domains and topology-ordered.
    """
    return RayVirtualCluster(
        bundle_ct_per_node_list=[spec.gpus_per_node] * spec.num_nodes,
        use_gpus=True,
        max_colocated_worker_groups=spec.max_colocated_worker_groups,
        num_gpus_per_node=spec.gpus_per_node,
        name=spec.name,
        placement_group_strategy=spec.placement_strategy,
        segment_size=spec.segment_size,
        node_resource_constraints=_segment_constraints(spec),
    )

def _build_sandbox_cluster(spec: FleetSpec) -> RayVirtualCluster:
    """CPU-only cluster; one bundle per concurrent episode slot.

    Sandbox containers run coding tasks: shell commands, compilation,
    test execution.  No GPU resources are claimed so sandbox actors never
    block inference fleet GPU allocation.

    Each bundle in the placement group maps to one concurrent episode
    slot.  bundle_ct_per_node_list is a single-element list whose value
    equals the total number of slots across all sandbox nodes, which lets
    Ray distribute them freely across CPU capacity.
    """
    total_slots = spec.num_nodes  # caller sets num_nodes = desired concurrency
    return RayVirtualCluster(
        bundle_ct_per_node_list=[total_slots],
        use_gpus=False,
        max_colocated_worker_groups=spec.max_colocated_worker_groups,
        num_gpus_per_node=0,
        name=spec.name,
        placement_group_strategy=spec.placement_strategy,
    )