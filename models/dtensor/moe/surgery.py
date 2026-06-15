"""HF MoE model surgery: swap routed-expert MLPs for the native ``MoEBlock``.

Replaces each routed-MoE layer of a loaded HF model with the native
``MoEBlock`` (``TokenChoiceTopKRouter`` + grouped-GEMM ``GroupedExperts``), so
``is_moe_state_dict`` fires, the v2 worker's ``_parallelize`` finds
``GroupedExperts`` to EP-shard, and the EP-aware refit path activates. Arch-keyed
registry — Qwen3-MoE (e.g. Qwen3-30B-A3B) first; GLM is a later entry.

LAYOUT NOTE (transformers 5.x). Modern ``transformers`` stores Qwen3-MoE experts
as FUSED 3D params on the loaded module, not per-expert Linears: the on-disk
per-expert checkpoint (``mlp.experts.{i}.{gate,up,down}_proj``) is merged at load
(``conversion_mapping.py``) into ``Qwen3MoeExperts.gate_up_proj`` ``(E, 2F, D)``
(gate||up concatenated on dim 1) and ``down_proj`` ``(E, D, F)``. So surgery
restructures those fused tensors directly into the native ``w1/w2/w3_EFD`` — this
is NOT ``convert.py``'s per-expert path (which still serves the on-disk/round-trip
and per-expert-Linear archs). The mapping is pure slicing:

    w1_EFD (gate) = gate_up_proj[:, :F, :]     # silu branch
    w3_EFD (up)   = gate_up_proj[:, F:, :]
    w2_EDF (down) = down_proj

matching ``GroupedExperts._experts_forward`` (``silu(x@w1ᵀ)*x@w3ᵀ -> @w2ᵀ``)
against Qwen3's ``act_fn(gate)*up -> down``.

The surgery PLAN (module-tree shape + fused param names + the gate/up split) is
CPU-validated on a tiny config (``test_moe_surgery.py``). The live forward
(grouped-GEMM, CUDA-only), FSDP wrap, and 30B weight load are device-bound
(hardware-deferred-validation.md, HV-15).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn

from dockyard_rl.models.dtensor.moe.config import build_token_dispatcher
from dockyard_rl.models.dtensor.moe.experts import GroupedExperts
from dockyard_rl.models.dtensor.moe.block import MoEBlock
from dockyard_rl.models.dtensor.moe.router import TokenChoiceTopKRouter


def _as_expert_param(t: torch.Tensor) -> nn.Parameter:
    """Detach from any HF autograd graph and make contiguous for grouped-GEMM."""
    return nn.Parameter(t.detach().contiguous())


def _build_moe_block_from_qwen3(
    sparse_block: nn.Module,
    dtensor_cfg: Any,
    load_balance_coeff: Optional[float],
) -> MoEBlock:
    """Build a native ``MoEBlock`` from a ``Qwen3MoeSparseMoeBlock``."""
    # nn.Module.__getattr__ is typed Tensor | Module; these are concrete
    # submodules — read their attrs through Any-typed handles.
    hf_experts: Any = sparse_block.experts  # Qwen3MoeExperts
    hf_gate: Any = sparse_block.gate        # Qwen3MoeTopKRouter

    num_experts = int(hf_experts.num_experts)
    hidden_dim = int(hf_experts.hidden_dim)         # D (model dim)
    intermediate = int(hf_experts.intermediate_dim)  # F (moe_intermediate_size)
    top_k = int(hf_gate.top_k)
    norm_topk_prob = bool(hf_gate.norm_topk_prob)

    router = TokenChoiceTopKRouter(
        dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        score_func="softmax",
        route_norm=norm_topk_prob,
        route_scale=1.0,
    )
    # HF router gate weight (E, D) -> nn.Linear(D, E).weight is also (E, D).
    router.gate.weight = nn.Parameter(hf_gate.weight.detach().clone())

    # ep=1 path: the local expert count equals the full count (alltoall/ep>1 is
    # rejected by build_token_dispatcher until HV-10 lands).
    dispatcher = build_token_dispatcher(
        dtensor_cfg, num_experts=num_experts, top_k=top_k
    )
    experts = GroupedExperts(
        dim=hidden_dim,
        hidden_dim=intermediate,
        num_experts=num_experts,
        dispatcher=dispatcher,
    )

    gate_up = hf_experts.gate_up_proj  # (E, 2F, D), gate||up on dim 1
    experts.w1_EFD = _as_expert_param(gate_up[:, :intermediate, :])  # gate
    experts.w3_EFD = _as_expert_param(gate_up[:, intermediate:, :])  # up
    experts.w2_EDF = _as_expert_param(hf_experts.down_proj)          # down

    return MoEBlock(
        router=router,
        experts=experts,
        num_experts=num_experts,
        load_balance_coeff=load_balance_coeff,
    )


def _surgery_qwen3_moe(
    model: nn.Module, dtensor_cfg: Any, *, load_balance_coeff: Optional[float]
) -> int:
    """Replace every ``Qwen3MoeSparseMoeBlock`` (a decoder layer's ``.mlp``)."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock,
    )

    count = 0
    for module in model.modules():
        mlp = getattr(module, "mlp", None)
        if isinstance(mlp, Qwen3MoeSparseMoeBlock):
            module.mlp = _build_moe_block_from_qwen3(
                mlp, dtensor_cfg, load_balance_coeff
            )
            count += 1
    if count == 0:
        raise ValueError(
            "no Qwen3MoeSparseMoeBlock layers found; the model is not a routed "
            "Qwen3-MoE checkpoint (dense layers have a plain Qwen3MoeMLP)."
        )
    return count


# HF architecture (model class name) -> surgery function. Keyed by name to avoid
# importing arch-specific transformers modules at import time (deferred inside).
_SURGERY_REGISTRY: dict[
    str, Callable[..., int]
] = {
    "Qwen3MoeForCausalLM": _surgery_qwen3_moe,
}


def is_moe_surgery_supported(model: nn.Module) -> bool:
    """Whether ``model``'s architecture has a registered MoE-surgery function.

    Lets a caller (the v2 worker's surgery seam) skip dense / unsupported models
    without provoking the ``NotImplementedError`` ``apply_moe_surgery`` raises.
    """
    return type(model).__name__ in _SURGERY_REGISTRY


def apply_moe_surgery(
    model: nn.Module,
    dtensor_cfg: Any,
    *,
    load_balance_coeff: Optional[float] = None,
) -> int:
    """Swap an HF MoE model's routed-expert MLPs for native ``MoEBlock`` in place.

    Args:
        model: A loaded HF causal-LM whose architecture is in the registry.
        dtensor_cfg: Drives the token dispatcher (``moe_parallelizer``).
        load_balance_coeff: If set, enables the aux-loss-free routing bias on
            each swapped block (its updater is the B.5b optimizer pre-hook).

    Returns:
        The number of layers converted.
    """
    arch = type(model).__name__
    fn = _SURGERY_REGISTRY.get(arch)
    if fn is None:
        raise NotImplementedError(
            f"MoE surgery is not implemented for architecture {arch!r}; "
            f"supported: {sorted(_SURGERY_REGISTRY)}."
        )
    return fn(model, dtensor_cfg, load_balance_coeff=load_balance_coeff)
