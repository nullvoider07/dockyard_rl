"""dockyard_rl.cluster — Ray cluster bootstrap and fleet management."""

from dockyard_rl.cluster.bootstrap import (
    FLEET_INFERENCE,
    FLEET_SANDBOX,
    FLEET_TRAINER,
    get_fleet_role,
    init_ray,
)
from dockyard_rl.cluster.fleet import (
    FleetSpec,
    build_cluster,
)
from dockyard_rl.cluster.placement import (
    ParallelismSpec,
    PlacementConfigError,
    inference_bundle_ct_per_node,
    inference_sync_ranks,
    nccl_sync_world_size,
    requires_unified_placement_group,
    trainer_bundle_ct_per_node,
    trainer_sync_ranks,
    validate_for_inference,
    validate_for_trainer,
)

__all__ = [
    # Fleet roles
    "FLEET_TRAINER",
    "FLEET_INFERENCE",
    "FLEET_SANDBOX",
    # Bootstrap
    "get_fleet_role",
    "init_ray",
    # Fleet management
    "FleetSpec",
    "build_cluster",
    # Placement helpers
    "ParallelismSpec",
    "PlacementConfigError",
    "validate_for_trainer",
    "validate_for_inference",
    "trainer_bundle_ct_per_node",
    "inference_bundle_ct_per_node",
    "requires_unified_placement_group",
    "nccl_sync_world_size",
    "trainer_sync_ranks",
    "inference_sync_ranks",
]