"""Placement group helpers for multi-fleet, multi-parallelism configurations.

Responsibilities:
  - Validate that a requested parallelism configuration fits within the
    declared fleet topology before any placement groups are created.
  - Compute the bundle_ct_per_node_list that VllmGeneration and the
    trainer's worker-group builders expect.
  - Determine whether a model's parallelism requires a unified cross-node
    placement group (TP > gpus_per_node) or per-node groups (TP <= g/n).

Nothing here creates Ray objects.  All functions are pure computation so
they can be called before init_ray() for pre-flight validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Parallelism specification
@dataclass(frozen=True)
class ParallelismSpec:
    """Requested model parallelism for one fleet.

    For the trainer fleet, tp/pp/dp describe how the policy optimizer
    is sharded.  For the inference fleet, tp/pp describe the vLLM worker
    topology; dp maps to the number of independent vLLM DP replicas
    (each DP replica holds a full model copy).

    expert_parallel_size > 1 activates expert-parallel mode for MoE
    models.  vLLM requires ep_size % tp_size == 0.

    Attributes:
        tensor_parallel_size:   Number of GPUs sharing one model layer.
        pipeline_parallel_size: Number of pipeline stages.
        data_parallel_size:     Number of independent model replicas.
        expert_parallel_size:   Expert parallelism for MoE.  Must be a
                                multiple of tensor_parallel_size.
    """
    tensor_parallel_size:   int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size:     int = 1
    expert_parallel_size:   int = 1

    @property
    def model_parallel_size(self) -> int:
        """GPUs consumed by one model replica (TP × PP)."""
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def total_gpus(self) -> int:
        """Total GPUs consumed by all replicas."""
        return self.model_parallel_size * self.data_parallel_size

# Validation and bundle layout computation for trainer and inference fleets
class PlacementConfigError(Exception):
    """Raised when a parallelism configuration cannot be satisfied."""

def validate_for_trainer(
    spec: ParallelismSpec,
    num_nodes: int,
    gpus_per_node: int,
) -> None:
    """Assert that spec fits within the declared trainer fleet topology.

    Raises PlacementConfigError with a descriptive message on failure.
    """
    total_fleet_gpus = num_nodes * gpus_per_node
    if spec.total_gpus > total_fleet_gpus:
        raise PlacementConfigError(
            f"Trainer parallelism requires {spec.total_gpus} GPUs "
            f"(TP={spec.tensor_parallel_size} × PP={spec.pipeline_parallel_size} "
            f"× DP={spec.data_parallel_size}) but the fleet only has "
            f"{total_fleet_gpus} GPUs ({num_nodes} nodes × {gpus_per_node} GPUs)."
        )
    if spec.total_gpus % gpus_per_node != 0:
        raise PlacementConfigError(
            f"Trainer total GPUs ({spec.total_gpus}) is not a multiple of "
            f"gpus_per_node ({gpus_per_node}).  Uneven placement groups are "
            f"not supported for the trainer fleet."
        )

def validate_for_inference(
    spec: ParallelismSpec,
    num_nodes: int,
    gpus_per_node: int,
) -> None:
    """Assert that spec fits within the declared inference fleet topology.

    Raises PlacementConfigError with a descriptive message on failure.
    """
    total_fleet_gpus = num_nodes * gpus_per_node
    if spec.total_gpus > total_fleet_gpus:
        raise PlacementConfigError(
            f"Inference parallelism requires {spec.total_gpus} GPUs "
            f"(TP={spec.tensor_parallel_size} × PP={spec.pipeline_parallel_size} "
            f"× DP={spec.data_parallel_size}) but the inference fleet only has "
            f"{total_fleet_gpus} GPUs ({num_nodes} nodes × {gpus_per_node} GPUs)."
        )
    if spec.expert_parallel_size > 1:
        if spec.expert_parallel_size % spec.tensor_parallel_size != 0:
            raise PlacementConfigError(
                f"Expert parallelism requires EP % TP == 0. "
                f"Got EP={spec.expert_parallel_size}, "
                f"TP={spec.tensor_parallel_size}."
            )

# Bundle layout computation
def trainer_bundle_ct_per_node(
    spec: ParallelismSpec,
    num_nodes: int,
    gpus_per_node: int,
) -> list[int]:
    """Return bundle_ct_per_node_list for the trainer fleet.

    Each bundle holds exactly one GPU (one rank).  The list has one entry
    per node; its value is the number of GPUs (bundles) on that node.

    Validates spec before computing.
    """
    validate_for_trainer(spec, num_nodes, gpus_per_node)
    nodes_used = spec.total_gpus // gpus_per_node
    return [gpus_per_node] * nodes_used

def inference_bundle_ct_per_node(
    spec: ParallelismSpec,
    num_nodes: int,
    gpus_per_node: int,
) -> list[int]:
    """Return bundle_ct_per_node_list for the inference fleet.

    Same shape as the trainer version.  VllmGeneration uses this list
    when constructing RayVirtualCluster, then decides internally whether
    to use per-node or unified placement groups based on whether
    TP > gpus_per_node.

    Validates spec before computing.
    """
    validate_for_inference(spec, num_nodes, gpus_per_node)
    nodes_used = spec.total_gpus // gpus_per_node
    return [gpus_per_node] * nodes_used

# Cross-node detection and NCCL sync group computation
def requires_unified_placement_group(
    spec: ParallelismSpec,
    gpus_per_node: int,
) -> bool:
    """Return True when TP spans multiple nodes.

    When TP > gpus_per_node, a single model layer is sharded across more
    GPUs than exist on one node.  Ray's per-node PACK placement groups
    cannot express this topology; VllmGeneration must use a single
    unified placement group spanning all inference nodes.

    This function lets the caller detect the condition before constructing
    RayVirtualCluster so logs and error messages can be emitted early.
    """
    return spec.tensor_parallel_size > gpus_per_node

def nccl_sync_world_size(
    trainer_spec: ParallelismSpec,
    inference_spec: ParallelismSpec,
) -> int:
    """Total ranks in the trainer↔inference NCCL weight-sync communicator.

    The sync group contains:
      - One rank per trainer DP rank 0 (TP rank 0, PP rank 0 per DP group).
        These are the ranks that hold the authoritative parameter shards
        after the optimizer step.
      - One rank per inference DP leader (TP rank 0 per vLLM DP replica).

    Both sides must agree on this world_size when calling
    torch.distributed.init_process_group() for the sync communicator.
    """
    trainer_dp_size   = trainer_spec.data_parallel_size
    inference_dp_size = inference_spec.data_parallel_size
    return trainer_dp_size + inference_dp_size

def trainer_sync_ranks(trainer_spec: ParallelismSpec) -> list[int]:
    """Rank indices within the sync communicator assigned to trainer DP leaders.

    Trainer DP leaders take the first trainer_dp_size ranks in the
    communicator (ranks 0 … trainer_dp_size - 1).  Inference DP leaders
    take the remaining ranks.
    """
    return list(range(trainer_spec.data_parallel_size))

def inference_sync_ranks(
    trainer_spec: ParallelismSpec,
    inference_spec: ParallelismSpec,
) -> list[int]:
    """Rank indices within the sync communicator assigned to inference DP leaders.

    Inference DP leaders take ranks trainer_dp_size … world_size - 1.
    """
    offset = trainer_spec.data_parallel_size
    return list(range(offset, offset + inference_spec.data_parallel_size))