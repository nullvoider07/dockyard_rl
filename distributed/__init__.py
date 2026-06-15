"""dockyard_rl.distributed — Ray virtual cluster and worker group abstractions."""

from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.distributed.stateless_process_group import StatelessProcessGroup
from dockyard_rl.distributed.named_sharding import NamedSharding
from dockyard_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
    ResourceInsufficiencyError,
)
from dockyard_rl.distributed.worker_groups_utils import (
    get_nsight_config_if_pattern_matches,
    recursive_merge_options,
)
from dockyard_rl.distributed.worker_groups import (
    MultiWorkerFuture,
    RayWorkerBuilder,
    RayWorkerGroup,
)

__all__ = [
    # Data container
    "BatchedDataDict",
    # Sharding
    "NamedSharding",
    # Virtual cluster
    "ClusterConfig",
    "RayVirtualCluster",
    "ResourceInsufficiencyError",
    # Worker groups
    "MultiWorkerFuture",
    "RayWorkerBuilder",
    "RayWorkerGroup",
    # Utilities
    "recursive_merge_options",
    "get_nsight_config_if_pattern_matches",
]