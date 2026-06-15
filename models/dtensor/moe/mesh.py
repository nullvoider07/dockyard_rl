"""Device-mesh construction for MoE expert parallelism.

Builds the two logical views of a single physical device mesh that MoE training
needs. Design follows the torchtitan ParallelDims model (pytorch/torchtitan,
BSD-3-Clause), reimplemented natively for dockyard:

  - dense view  (dp_replicate, dp_shard, cp, tp): attention / norm / router /
    dense-MLP params — HSDP (replicate x shard) + context + tensor parallel.
  - sparse view (dp_replicate, efsdp, ep): routed-expert params — experts
    Shard(0) over ``ep``, FSDP over ``efsdp`` within the EP region.

Both views re-factor the SAME world ranks (one flat world mesh, two
``_unflatten`` calls), so they are consistent by construction. EP is carved
from the dp_shard*cp*tp budget:

    efsdp = dp_shard * cp * tp // ep

and the world-size constraint EXCLUDES ep (it is not an independent world
factor):

    dp_replicate * dp_shard * cp * tp == world_size

The pure factoring + rank->expert assignment (``MoEParallelDims``) has no
torch.distributed dependency and is unit-tested without a GPU.
``build_moe_meshes`` is the device-bound wrapper (needs a live process group).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MoEParallelDims:
    """Pure parallelism factoring for MoE training (no torch.distributed).

    Holds the parallel degrees and derives the dependent sizes + the
    rank->expert assignment. EP is carved from dp_shard*cp*tp; the world-size
    constraint is over the dense factors only.

    Args:
        world_size:   Total number of ranks.
        dp_replicate: HSDP replicate degree (DDP-style outer data parallel).
        dp_shard:     FSDP shard degree (inner data parallel).
        cp:           Context-parallel degree.
        tp:           Tensor-parallel degree.
        ep:           Expert-parallel degree (carved from dp_shard*cp*tp).
    """

    world_size: int
    dp_replicate: int = 1
    dp_shard: int = 1
    cp: int = 1
    tp: int = 1
    ep: int = 1

    def __post_init__(self) -> None:
        for name, d in (
            ("world_size", self.world_size),
            ("dp_replicate", self.dp_replicate),
            ("dp_shard", self.dp_shard),
            ("cp", self.cp),
            ("tp", self.tp),
            ("ep", self.ep),
        ):
            if d < 1:
                raise ValueError(f"{name} must be >= 1, got {d}")

        dense_product = self.dp_replicate * self.dp_shard * self.cp * self.tp
        if dense_product != self.world_size:
            raise ValueError(
                f"dp_replicate({self.dp_replicate}) * dp_shard({self.dp_shard}) * "
                f"cp({self.cp}) * tp({self.tp}) = {dense_product} != "
                f"world_size({self.world_size}). ep is carved from dp_shard*cp*tp, "
                "not an independent world factor."
            )

        ep_budget = self.dp_shard * self.cp * self.tp
        if ep_budget % self.ep != 0:
            raise ValueError(
                f"ep({self.ep}) must divide dp_shard*cp*tp({ep_budget}) so efsdp "
                "(FSDP degree within the EP region) is integral."
            )

    @property
    def dp(self) -> int:
        """Total data-parallel degree (replicate x shard)."""
        return self.dp_replicate * self.dp_shard

    @property
    def fsdp(self) -> int:
        """Dense FSDP degree (dp_shard folded with cp)."""
        return self.dp_shard * self.cp

    @property
    def efsdp(self) -> int:
        """FSDP degree within the EP region: dp_shard*cp*tp // ep."""
        return (self.dp_shard * self.cp * self.tp) // self.ep

    @property
    def enable_ep(self) -> bool:
        return self.ep > 1

    def ep_coord(self, global_rank: int) -> int:
        """EP-group coordinate of a global rank.

        The sparse mesh (dp_replicate, efsdp, ep) is row-major-unflattened from
        the flat world mesh, so ``ep`` is the innermost (fastest-varying) axis:
        consecutive ranks form one EP group, giving ``ep_coord = rank % ep``.
        """
        if not 0 <= global_rank < self.world_size:
            raise ValueError(
                f"global_rank {global_rank} out of range [0, {self.world_size})"
            )
        return global_rank % self.ep

    def num_local_experts(self, num_experts: int) -> int:
        """Experts owned by each EP rank (num_experts must divide by ep)."""
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")
        if num_experts % self.ep != 0:
            raise ValueError(
                f"num_experts({num_experts}) must be divisible by ep({self.ep})"
            )
        return num_experts // self.ep

    def experts_for_rank(self, global_rank: int, num_experts: int) -> list[int]:
        """Global expert indices owned by ``global_rank`` (Shard(0) over ep).

        Experts are partitioned contiguously across the ``ep`` axis; ranks that
        share an ep coordinate (different efsdp / dp_replicate) hold the same
        partition (expert weights are replicated across those axes before FSDP).
        """
        local = self.num_local_experts(num_experts)
        e = self.ep_coord(global_rank)
        return list(range(e * local, (e + 1) * local))

    @classmethod
    def from_dtensor_cfg(
        cls, dtensor_cfg: Any, world_size: int
    ) -> "MoEParallelDims":
        """Build from a dtensor_cfg mapping, deriving dp_shard to fill world.

        dp_shard is computed as world_size // (dp_replicate * cp * tp); the
        remaining degrees come from the config (defaulting to 1).
        """
        dp_replicate = dtensor_cfg.get("dp_replicate_size", 1)
        cp = dtensor_cfg.get("context_parallel_size", 1)
        tp = dtensor_cfg.get("tensor_parallel_size", 1)
        ep = dtensor_cfg.get("expert_parallel_size", 1)

        non_shard = dp_replicate * cp * tp
        if non_shard == 0 or world_size % non_shard != 0:
            raise ValueError(
                f"world_size({world_size}) not divisible by "
                f"dp_replicate*cp*tp({non_shard}); cannot derive dp_shard."
            )
        dp_shard = world_size // non_shard
        return cls(
            world_size=world_size,
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
            cp=cp,
            tp=tp,
            ep=ep,
        )


def build_moe_meshes(dims: MoEParallelDims, device_type: str = "cuda") -> dict:
    """Build the dense + sparse device-mesh views from one flat world mesh.

    Device-bound: requires a live process group (call inside a worker). Mirrors
    torchtitan's ``ParallelDims.build_mesh`` — one flat world mesh unflattened
    into a dense view and (when ep > 1) a sparse view.

    Returns a dict with:
      - "dense":  (dp_replicate, dp_shard, cp, tp) with a flattened "dp" axis.
      - "sparse": (dp_replicate, efsdp, ep) — only present when ep > 1.

    Note: degree-1 dimensions still create (trivial) process groups here; the
    fake-backend elision torchtitan applies is a bring-up optimization, not
    required for correctness.
    """
    from torch.distributed.device_mesh import init_device_mesh

    world_mesh = init_device_mesh(
        device_type, (dims.world_size,), mesh_dim_names=("world",)
    )

    dense_mesh = world_mesh._unflatten(
        0,
        (dims.dp_replicate, dims.dp_shard, dims.cp, dims.tp),
        ("dp_replicate", "dp_shard", "cp", "tp"),
    )
    # Flatten the two data-parallel axes into a single "dp" view for HSDP.
    dense_mesh[("dp_replicate", "dp_shard")]._flatten("dp")

    meshes = {"dense": dense_mesh}

    if dims.enable_ep:
        meshes["sparse"] = world_mesh._unflatten(
            0,
            (dims.dp_replicate, dims.efsdp, dims.ep),
            ("dp_replicate", "efsdp", "ep"),
        )

    return meshes
