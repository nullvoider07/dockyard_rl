"""Apply the MoE sharding policy to a GroupedExperts module (device-bound).

Wraps each routed-expert weight as a DTensor with the placements from
``sharding.py`` over the B.2 sparse mesh (``Shard(0)`` on ``ep``), and replicates
the router gate. This is the application of the B.3 policy; it needs a live
device mesh and is exercised only at bring-up (hardware-deferred-validation.md
HV-9). The placement DECISIONS it relies on are unit-tested in
``test_moe_sharding.py``; FSDP (``fully_shard``) over the ``efsdp`` axis is applied
separately by the v2 worker after this.
"""

from __future__ import annotations

from typing import Optional

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor

from dockyard_rl.models.dtensor.moe.experts import GroupedExperts
from dockyard_rl.models.dtensor.moe.sharding import (
    ExpertRole,
    routed_expert_placements,
    router_gate_placements,
)

# Routed-expert weight -> TP role (used only when EP is off; see sharding.py).
_EXPERT_WEIGHT_ROLES: dict[str, ExpertRole] = {
    "w1_EFD": "gate_up",
    "w2_EDF": "down",
    "w3_EFD": "gate_up",
}


def parallelize_grouped_experts(
    experts: GroupedExperts,
    mesh: DeviceMesh,
    *,
    enable_ep: bool,
) -> GroupedExperts:
    """Distribute the routed-expert weights over ``mesh`` per the B.3 policy.

    Args:
        experts: The GroupedExperts module to shard in place.
        mesh: Target mesh — the sparse mesh ``(dp_replicate, efsdp, ep)`` when
            ``enable_ep``, else the dense mesh ``(dp_replicate, dp_shard, cp, tp)``.
        enable_ep: Whether expert parallelism is active.

    Returns:
        The same module, with ``w1_EFD``/``w2_EDF``/``w3_EFD`` now DTensor params.
    """
    dim_names = tuple(mesh.mesh_dim_names or ())
    for name, role in _EXPERT_WEIGHT_ROLES.items():
        weight = getattr(experts, name)
        placements = routed_expert_placements(
            dim_names, enable_ep=enable_ep, role=role
        )
        setattr(
            experts,
            name,
            nn.Parameter(distribute_tensor(weight.data, mesh, placements)),
        )
    return experts


def parallelize_router_gate(
    gate: nn.Module,
    mesh: DeviceMesh,
    weight_attr: str = "weight",
    bias_attr: Optional[str] = "bias",
) -> nn.Module:
    """Replicate the router gate weight (and bias) over ``mesh``.

    The gate is tiny and every token needs the full routing logits, so it is
    replicated rather than sharded.
    """
    dim_names = tuple(mesh.mesh_dim_names or ())
    placements = router_gate_placements(dim_names)

    weight = getattr(gate, weight_attr)
    setattr(gate, weight_attr, nn.Parameter(distribute_tensor(weight.data, mesh, placements)))

    if bias_attr is not None and getattr(gate, bias_attr, None) is not None:
        bias = getattr(gate, bias_attr)
        setattr(gate, bias_attr, nn.Parameter(distribute_tensor(bias.data, mesh, placements)))

    return gate
