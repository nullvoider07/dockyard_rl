"""MoE expert-parallelism support for the DTensor policy stack.

Native dockyard implementation of the device-mesh topology and (later) the
expert sharding / grouped-GEMM compute for mixture-of-experts training.
"""

from dockyard_rl.models.dtensor.moe.mesh import (
    MoEParallelDims,
    build_moe_meshes,
)
from dockyard_rl.models.dtensor.moe.convert import (
    fuse_hf_expert_state_dict,
    is_hf_per_expert_key,
)
from dockyard_rl.models.dtensor.moe.detect import is_moe_state_dict
from dockyard_rl.models.dtensor.moe.dispatch import (
    AllToAllDispatchMetadata,
    AllToAllTokenDispatcher,
    LocalDispatchMetadata,
    LocalTokenDispatcher,
)
from dockyard_rl.models.dtensor.moe.block import (
    MoEBlock,
    routing_map_and_counts,
)
from dockyard_rl.models.dtensor.moe.experts import GroupedExperts
from dockyard_rl.models.dtensor.moe.router import (
    MISSING_ROUTE_SENTINEL,
    TokenChoiceTopKRouter,
)
from dockyard_rl.models.dtensor.moe.router_replay import (
    bind_router_replay,
    count_moe_blocks,
    iter_moe_blocks,
    resolve_router_replay_enabled,
    router_replay_context,
    validate_router_replay_config,
)
from dockyard_rl.models.dtensor.moe.parallelize_moe import (
    parallelize_grouped_experts,
    parallelize_router_gate,
)
from dockyard_rl.models.dtensor.moe.refit import (
    expand_grouped_expert_info,
    expand_grouped_expert_key,
    is_grouped_expert_key,
    iter_expanded_refit_tensors,
)
from dockyard_rl.models.dtensor.moe.sharding import (
    num_local_experts,
    routed_expert_placements,
    router_gate_placements,
)
from dockyard_rl.models.dtensor.moe.config import (
    build_token_dispatcher,
    read_num_routed_experts,
    resolve_moe_parallelizer,
    validate_moe_parallel_config,
)
from dockyard_rl.models.dtensor.moe.load_balance import (
    compute_expert_bias_delta,
    register_expert_bias_update_hook,
    update_expert_biases,
)
from dockyard_rl.models.dtensor.moe.surgery import (
    apply_moe_surgery,
    is_moe_surgery_supported,
)

__all__ = [
    "MoEParallelDims",
    "build_moe_meshes",
    "is_moe_state_dict",
    "fuse_hf_expert_state_dict",
    "is_hf_per_expert_key",
    "num_local_experts",
    "routed_expert_placements",
    "router_gate_placements",
    "AllToAllDispatchMetadata",
    "AllToAllTokenDispatcher",
    "LocalDispatchMetadata",
    "LocalTokenDispatcher",
    "GroupedExperts",
    "MoEBlock",
    "routing_map_and_counts",
    "TokenChoiceTopKRouter",
    "MISSING_ROUTE_SENTINEL",
    "bind_router_replay",
    "count_moe_blocks",
    "iter_moe_blocks",
    "resolve_router_replay_enabled",
    "router_replay_context",
    "validate_router_replay_config",
    "parallelize_grouped_experts",
    "parallelize_router_gate",
    "expand_grouped_expert_info",
    "expand_grouped_expert_key",
    "is_grouped_expert_key",
    "iter_expanded_refit_tensors",
    "build_token_dispatcher",
    "read_num_routed_experts",
    "resolve_moe_parallelizer",
    "validate_moe_parallel_config",
    "compute_expert_bias_delta",
    "register_expert_bias_update_hook",
    "update_expert_biases",
    "apply_moe_surgery",
    "is_moe_surgery_supported",
]
