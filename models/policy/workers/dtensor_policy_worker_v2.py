"""MoE-capable DTensor policy worker (v2).

Extends ``DTensorPolicyWorkerImpl`` with native expert parallelism. It overrides
only the two seams the v1 refactor exposed:

  - ``_build_device_mesh`` — for the dense, non-HSDP case (ep=1, dp_replicate=1)
    it defers to v1 verbatim (byte-identical bring-up). When expert parallelism
    or HSDP is requested it builds one flat world mesh and derives both the dense
    (dp_replicate, dp_shard, cp, tp) handles v1 consumes and the sparse
    (dp_replicate, efsdp, ep) mesh the routed experts shard over.
  - ``_parallelize`` — applies the B.3 expert-sharding policy to every
    ``GroupedExperts`` submodule (over the sparse mesh) before the dense FSDP/TP
    plan covers the remaining (attention / norm / router) parameters.

The MoE mesh construction and expert sharding are device-bound (live process
group + CUDA grouped-GEMM); they are exercised at bring-up and tracked in
hardware-deferred-validation.md (HV-2/HV-9). The pure decisions they rely on
(mesh topology math, placement policy, dispatch) are unit-tested no-GPU.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, cast

import ray
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from dockyard_rl.models.dtensor.moe import (
    AllToAllTokenDispatcher,
    GroupedExperts,
    MoEParallelDims,
    apply_moe_surgery,
    build_moe_meshes,
    expand_grouped_expert_info,
    is_grouped_expert_key,
    is_moe_state_dict,
    is_moe_surgery_supported,
    iter_expanded_refit_tensors,
    parallelize_grouped_experts,
    register_expert_bias_update_hook,
    resolve_moe_parallelizer,
)
from dockyard_rl.models.policy.workers.dtensor_policy_worker import (
    DTensorPolicyWorkerImpl,
    get_runtime_env_for_policy_worker,
)


class DTensorPolicyWorkerV2Impl(DTensorPolicyWorkerImpl):  # type: ignore[misc]
    """DTensor policy worker with native MoE expert parallelism."""

    moe_dims: MoEParallelDims
    moe_mesh: Optional[DeviceMesh]

    def _apply_moe_surgery(self, model: nn.Module) -> None:
        """Swap HF routed-expert MLPs for native ``MoEBlock`` modules (B.5a).

        Skips dense / unsupported architectures (the v1 behavior — byte-identical
        model build). For a supported MoE arch it runs ``apply_moe_surgery``,
        threading the ``load_balance_coeff`` and ``token_dispatcher`` from the
        single ``moe_parallelizer`` config seam. Called by the v1 model-build
        path on both the rank-0 source model (so its state dict broadcasts native
        param names) and the empty per-rank model (so keys match). The live
        grouped-GEMM forward / FSDP wrap / 30B load are device-bound (HV-15).
        """
        if not is_moe_surgery_supported(model):
            return
        dtensor_cfg = self.cfg["dtensor_cfg"]
        parallelizer = resolve_moe_parallelizer(dtensor_cfg)
        converted = apply_moe_surgery(
            model,
            dtensor_cfg,
            load_balance_coeff=parallelizer["load_balance_coeff"],
        )
        print(
            f"[Rank {self.rank}] MoE surgery: converted {converted} routed-MoE "
            f"layer(s) to native MoEBlock (load_balance_coeff="
            f"{parallelizer['load_balance_coeff']})."
        )

    def _build_device_mesh(
        self, dp_size: int, cp_size: int, tp_size: int
    ) -> None:
        """Build the dense mesh (and, when MoE/HSDP, the sparse expert mesh).

        For ep=1 and dp_replicate=1 the dense mesh is exactly v1's, so this
        defers to ``super()`` and leaves ``moe_mesh`` unset.
        """
        world_size = torch.distributed.get_world_size()
        self.moe_dims = MoEParallelDims.from_dtensor_cfg(
            self.cfg["dtensor_cfg"], world_size
        )

        if not self.moe_dims.enable_ep and self.moe_dims.dp_replicate == 1:
            super()._build_device_mesh(dp_size, cp_size, tp_size)
            self.moe_mesh = None
            return

        self._build_moe_device_mesh(cp_size, tp_size)

    def _build_moe_device_mesh(self, cp_size: int, tp_size: int) -> None:
        """Device-bound MoE/HSDP mesh build (HV-2).

        Builds one flat world mesh, unflattens the dense
        (dp_replicate, dp_shard, cp, tp) view and the sparse
        (dp_replicate, efsdp, ep) view, then maps the dense handles onto the
        attributes v1's training/parallelize code reads (``device_mesh``,
        ``dp_mesh``, ``tp_mesh``, ``cp_mesh``, ``dp_cp_mesh``, ``dp_size`` ...).
        ``dp_mesh`` is the flattened ``dp`` axis (dp_replicate * dp_shard),
        matching v1's ``dp_size = world // tp // cp``.
        """
        meshes = build_moe_meshes(self.moe_dims, device_type="cuda")
        dense = meshes["dense"]

        self.device_mesh = dense
        self.tp_mesh = dense["tp"]
        self.cp_mesh = dense["cp"]
        self.dp_mesh = dense["dp"]  # flattened dp_replicate * dp_shard
        # FSDP shards over the data + context-parallel axes; flatten them into a
        # single mesh the same way v1 does, off the already-flattened dp axis.
        self.dp_cp_mesh = dense[("dp", "cp")]._flatten(mesh_dim_name="dp_cp")

        self.dp_size = self.moe_dims.dp_replicate * self.moe_dims.dp_shard
        self.tp_size = tp_size
        self.cp_size = cp_size

        self.moe_mesh = meshes.get("sparse")

    def _parallelize(self, sequence_parallel_enabled: bool) -> None:
        """Shard routed experts over the sparse mesh, then apply the dense plan.

        For a non-MoE model this is exactly v1. For a MoE model, every
        ``GroupedExperts`` submodule is distributed via the B.3 policy first
        (so its weights are already DTensors when the dense FSDP/TP plan runs),
        then ``super()._parallelize`` covers attention / norm / router.

        HV-12 (ep>1): the dense ``_parallelize_model`` ``fully_shard``s each layer
        over ``dp_cp`` WITHOUT excluding the routed experts, which are DTensors on
        the sparse mesh (``Shard(0)`` on ep, ``Replicate`` on efsdp). torchtitan
        instead FSDPs experts separately over the efsdp axis via a unified
        ``fully_shard(block, shard_placement_fn=...)`` on an edp_mesh with a
        Shard(0)/Shard(1) padding choice — a composition that diverges from this
        pre-``distribute_tensor`` approach and is device-bound (no CPU validation
        of FSDP/mesh composition). It is resolved at ep>1 bring-up; the ep=1 path
        (experts on the dense mesh) is unaffected.
        """
        state_dict_keys = self.model.state_dict().keys()
        if not is_moe_state_dict(state_dict_keys):
            super()._parallelize(sequence_parallel_enabled)
            return

        expert_mesh = self.moe_mesh if self.moe_mesh is not None else self.device_mesh
        enable_ep = self.moe_dims.enable_ep

        sharded = 0
        for module in self.model.modules():
            if isinstance(module, GroupedExperts):
                parallelize_grouped_experts(
                    module, expert_mesh, enable_ep=enable_ep
                )
                # Wire the cross-rank all-to-all dispatcher's EP submesh (HV-10).
                # ep=1 leaves it unwired (local fallback); enable_ep implies the
                # sparse mesh exists (built in _build_moe_device_mesh).
                if enable_ep and self.moe_mesh is not None and isinstance(
                    module.dispatcher, AllToAllTokenDispatcher
                ):
                    module.dispatcher.wire_ep_mesh(self.moe_mesh["ep"])
                sharded += 1

        if sharded == 0:
            # MoE checkpoint detected but no native GroupedExperts modules: the
            # loaded HF model still uses per-expert Linear submodules
            # (mlp.experts.N.*_proj). Converting those into GroupedExperts is a
            # separate model-integration step; without it the experts fall to
            # the dense FSDP/TP plan below (correct but not expert-parallel).
            print(
                f"[Rank {self.rank}] MoE state dict detected but no native "
                "GroupedExperts modules found; routed experts will use the "
                "dense parallel plan (native-expert conversion not applied)."
            )

        super()._parallelize(sequence_parallel_enabled)

    def _iter_refit_state_dict(self) -> Iterable[tuple[str, torch.Tensor]]:
        """Stream native experts as per-expert HF tensors (EP-layout reshard).

        Fused GroupedExperts params (``w1_EFD`` ...) are EP-sharded; the
        inference backend loads per-expert HF tensors. Expand them: gather over
        EP (``full_tensor`` — device-bound, HV-13) then unbind dim 0 into
        ``...experts.{i}.{gate,up,down}_proj.weight``. Non-expert params pass
        through unchanged for the transport's own materialization. For a model
        with no native GroupedExperts (dense / raw-HF MoE) this is identical to
        v1 (no key matches).
        """
        return cast(
            Iterable[tuple[str, torch.Tensor]],
            iter_expanded_refit_tensors(
                self.model.state_dict().items(),
                materialize=lambda t: t.full_tensor() if isinstance(t, DTensor) else t,
            ),
        )

    def prepare_refit_info(
        self, state_dict_info: Optional[dict[str, Any]] = None
    ) -> Optional[dict[str, Any]]:
        """Refit metadata with fused-expert entries expanded per-expert.

        Mirrors ``_iter_refit_state_dict`` on the metadata side so the inference
        backend's name->(shape, dtype) map matches the streamed tensors.
        """
        result: dict[str, Any] = {}
        for name, tensor in self.model.state_dict().items():
            if is_grouped_expert_key(name):
                for exp_name, exp_shape in expand_grouped_expert_info(
                    name, tuple(tensor.shape)
                ):
                    result[exp_name] = (torch.Size(exp_shape), self.dtype)
            else:
                result[name] = (tensor.shape, self.dtype)
        return result

    def _setup_moe_load_balancing(self) -> None:
        """Register the aux-loss-free expert-bias updater (B.5b).

        No-op unless the model has load-balanced ``MoEBlock`` modules (i.e. the
        HF→MoEBlock surgery ran with ``load_balance_coeff`` set). The cross-rank
        token-load reduction is injected device-side (HV-14).
        """
        if self.optimizer is None:
            return
        registered = register_expert_bias_update_hook(
            self.optimizer,
            self.model,
            reduce_tokens_fn=self._reduce_tokens_per_expert,
        )
        if registered:
            print(
                f"[Rank {self.rank}] MoE aux-loss-free load-balancing hook "
                "registered (expert-bias updater)."
            )

    def _reduce_tokens_per_expert(
        self, tokens_per_expert_E: torch.Tensor
    ) -> torch.Tensor:
        """Sum per-expert token counts over the data complement (device-bound).

        Each rank counts only its local token shard, so the global load is the
        SUM over the data+context-parallel ranks (``dp_cp``); reducing there
        makes every rank's ``expert_bias_E`` step identical. EP-axis handling and
        whether the tp-replicated counts need exclusion are confirmed at bring-up
        (HV-14).
        """
        if not torch.distributed.is_initialized():
            return tokens_per_expert_E
        reduced = tokens_per_expert_E.clone()
        torch.distributed.all_reduce(
            reduced,
            op=torch.distributed.ReduceOp.SUM,
            group=self.dp_cp_mesh.get_group(),
        )
        return reduced


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("dtensor_policy_worker")
)  # pragma: no cover
class DTensorPolicyWorkerV2(DTensorPolicyWorkerV2Impl):
    pass
