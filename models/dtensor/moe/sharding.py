"""Expert sharding placement policy for MoE blocks.

Distilled from torchtitan's ``moe_sharding.py`` (pytorch/torchtitan, BSD-3) into
dockyard's plain-DTensor idiom — WITHOUT torchtitan's sharding framework
(``ShardingConfig`` / ``NamedPlacement`` / ``MeshAxisName`` / ``decoder_sharding``).
This module encodes only the placement DECISIONS: which DTensor placement each
MoE parameter category takes, ordered for a given device mesh's dims.

The device-bound APPLICATION (``distribute_tensor`` over the B.2 sparse mesh /
``parallelize_module`` for the router + shared experts) lives with the
``GroupedExperts`` module that B.4 vendors — it cannot be written or tested
against a phantom module, and is exercised only at bring-up (see
``hardware-deferred-validation.md``).

Param categories (torchtitan GroupedExperts / MoE module structure):
  - routed experts: ``experts.{w1_EFD, w2_EDF, w3_EFD}`` — 3D ``(num_experts, *, *)``;
    EP on  -> ``Shard(0)`` over ``ep`` (sparse mesh);
    EP off -> dense TP shard on the ``tp`` axis (colwise gate/up, rowwise down).
  - router gate: ``router.gate.weight`` — ``Replicate`` on every axis.
  - shared experts: a normal dense FFN — use the standard Colwise/Rowwise
    ``ParallelStyle`` in the TP plan (no custom placement here; see notes).
"""

from __future__ import annotations

from typing import Literal, Sequence

from torch.distributed.tensor.placement_types import Placement, Replicate, Shard

# Routed-expert weight roles. Only relevant when EP is OFF (dense TP sharding):
#   gate_up (w1_EFD / w3_EFD): colwise -> shard the output (hidden) dim -> Shard(1)
#   down    (w2_EDF):          rowwise -> shard the input  (hidden) dim -> Shard(2)
ExpertRole = Literal["gate_up", "down"]

_DENSE_TP_SHARD: dict[ExpertRole, Placement] = {
    "gate_up": Shard(1),
    "down": Shard(2),
}


def routed_expert_placements(
    mesh_dim_names: Sequence[str],
    *,
    enable_ep: bool,
    role: ExpertRole,
) -> list[Placement]:
    """Ordered DTensor placements for one routed-expert weight.

    Args:
        mesh_dim_names: The target mesh's dim names, in order. With EP this is
            the sparse mesh ``("dp_replicate", "efsdp", "ep")``; without EP it is
            the dense mesh ``("dp_replicate", "dp_shard", "cp", "tp")``.
        enable_ep: Whether expert parallelism is active (ep > 1).
        role: ``gate_up`` (w1/w3) or ``down`` (w2) — selects the TP shard dim
            when EP is off. Ignored when EP is on (experts always ``Shard(0)``).

    Returns:
        A placement per mesh dim. EP on: ``Shard(0)`` on ``ep``, ``Replicate``
        elsewhere. EP off: ``Shard(1|2)`` on ``tp``, ``Replicate`` elsewhere.
    """
    placements: list[Placement] = []
    for name in mesh_dim_names:
        if enable_ep:
            placements.append(Shard(0) if name == "ep" else Replicate())
        else:
            placements.append(
                _DENSE_TP_SHARD[role] if name == "tp" else Replicate()
            )
    return placements


def router_gate_placements(mesh_dim_names: Sequence[str]) -> list[Placement]:
    """Router gate weight is replicated on every mesh axis.

    Every token needs the full set of routing logits, so the gate (tiny) is not
    sharded; it is replicated and computed locally.
    """
    return [Replicate() for _ in mesh_dim_names]


def num_local_experts(num_experts: int, ep_size: int) -> int:
    """Experts stored per EP rank (``num_experts`` must divide ``ep_size``).

    Mirrors ``MoEParallelDims.num_local_experts``; provided here so the sharding
    layer can size the local expert tensor without importing the mesh module.
    """
    if num_experts <= 0:
        raise ValueError(f"num_experts must be > 0, got {num_experts}")
    if ep_size <= 0:
        raise ValueError(f"ep_size must be > 0, got {ep_size}")
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts({num_experts}) must be divisible by ep_size({ep_size})"
        )
    return num_experts // ep_size
