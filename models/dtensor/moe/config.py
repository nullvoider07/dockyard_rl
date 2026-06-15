"""MoE/EP config surface: resolution, validation, and dispatcher selection.

The single place that reads the ``dtensor_cfg.moe_parallelizer`` block and the
expert-parallel degrees, so the rest of the stack does not re-parse config:

  - ``resolve_moe_parallelizer`` — defaults + value validation for
    ``{token_dispatcher, grouped_gemm}``.
  - ``validate_moe_parallel_config`` — config-level (early, driver-side) check of
    the world-size factoring and the dispatcher<->ep consistency, reusing
    ``MoEParallelDims`` (which raises on bad divisibility).
  - ``build_token_dispatcher`` — the factory the model-surgery path calls to
    construct the routed-token dispatcher; ``alltoall`` raises a clear
    ``NotImplementedError`` (HV-10) until the cross-rank dispatcher is built.
  - ``read_num_routed_experts`` — best-effort routed-expert count from a model's
    HF config (deferred ``transformers`` import), so the ``num_experts % ep``
    check can fire at policy setup instead of as a cryptic mesh failure later.

The ``moe_parallelizer`` degrees here are the TRAINER-side expert parallelism;
they are distinct from the inference-side ``generation.vllm_cfg.expert_parallel_size``.
"""

from __future__ import annotations

from typing import Any, Optional

from dockyard_rl.models.dtensor.moe.dispatch import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)
from dockyard_rl.models.dtensor.moe.mesh import MoEParallelDims

DEFAULT_TOKEN_DISPATCHER = "local"
_VALID_TOKEN_DISPATCHERS = ("local", "alltoall")

# HF config attributes that carry the total routed-expert count, by convention
# across architectures (checked in order; first positive int wins).
_ROUTED_EXPERT_COUNT_ATTRS = (
    "num_experts",        # Qwen3-MoE
    "n_routed_experts",   # DeepSeek-V2/V3
    "num_local_experts",  # Mixtral (total count despite the name)
    "moe_num_experts",    # GLM MoE variants
)


def resolve_moe_parallelizer(dtensor_cfg: Any) -> dict[str, Any]:
    """Read and validate the ``moe_parallelizer`` block, applying defaults.

    Returns ``{"token_dispatcher": str, "grouped_gemm": True,
    "load_balance_coeff": Optional[float]}``. Defaults reproduce the dense path
    (local dispatch, grouped-GEMM compute, load balancing off). Rejects an
    unknown dispatcher, ``grouped_gemm: false`` (the per-expert-loop compute path
    is out of scope), and a non-positive ``load_balance_coeff`` with a
    config-level error.
    """
    raw = dtensor_cfg.get("moe_parallelizer") or {}
    token_dispatcher = raw.get("token_dispatcher", DEFAULT_TOKEN_DISPATCHER)
    grouped_gemm = raw.get("grouped_gemm", True)
    load_balance_coeff = raw.get("load_balance_coeff", None)

    if token_dispatcher not in _VALID_TOKEN_DISPATCHERS:
        raise ValueError(
            f"moe_parallelizer.token_dispatcher={token_dispatcher!r} is invalid; "
            f"expected one of {_VALID_TOKEN_DISPATCHERS}."
        )
    if grouped_gemm is not True:
        raise ValueError(
            "moe_parallelizer.grouped_gemm=false is not implemented; routed-expert "
            "compute is grouped-GEMM only (a per-expert loop is out of scope). "
            "Set grouped_gemm: true."
        )
    if load_balance_coeff is not None:
        if not isinstance(load_balance_coeff, (int, float)) or isinstance(
            load_balance_coeff, bool
        ):
            raise ValueError(
                "moe_parallelizer.load_balance_coeff must be a number or null, got "
                f"{load_balance_coeff!r}."
            )
        if load_balance_coeff <= 0:
            raise ValueError(
                "moe_parallelizer.load_balance_coeff must be positive (the "
                f"aux-loss-free bias step size), got {load_balance_coeff}. Use null "
                "to disable load balancing."
            )
        load_balance_coeff = float(load_balance_coeff)
    return {
        "token_dispatcher": token_dispatcher,
        "grouped_gemm": True,
        "load_balance_coeff": load_balance_coeff,
    }


def validate_moe_parallel_config(
    dtensor_cfg: Any, world_size: int, num_experts: Optional[int] = None
) -> MoEParallelDims:
    """Validate the MoE/EP config early (driver-side), returning the factoring.

    Surfaces config-level errors before any device-mesh build:
      - ``dp_replicate*cp*tp`` divides ``world_size`` (via ``from_dtensor_cfg``);
      - ``token_dispatcher: local`` is only valid for ``expert_parallel_size=1``;
      - when ``num_experts`` is known, ``num_experts % ep == 0`` (via
        ``num_local_experts``).
    """
    parallelizer = resolve_moe_parallelizer(dtensor_cfg)
    dims = MoEParallelDims.from_dtensor_cfg(dtensor_cfg, world_size)

    if parallelizer["token_dispatcher"] == "local" and dims.enable_ep:
        raise ValueError(
            "moe_parallelizer.token_dispatcher='local' supports only "
            f"expert_parallel_size=1, but expert_parallel_size={dims.ep}. Set "
            "token_dispatcher='alltoall' for ep>1 (cross-rank routing; HV-10)."
        )

    if num_experts is not None:
        dims.num_local_experts(num_experts)  # raises if num_experts % ep != 0
    return dims


def build_token_dispatcher(
    dtensor_cfg: Any,
    *,
    num_experts: int,
    top_k: int,
    score_before_experts: bool = True,
) -> LocalTokenDispatcher:
    """Construct the routed-token dispatcher selected by ``moe_parallelizer``.

    ``num_experts`` is the GLOBAL routed-expert count; experts are partitioned
    contiguously across EP ranks (see ``mesh.experts_for_rank``). ``local``
    returns a ``LocalTokenDispatcher`` (ep=1). ``alltoall`` returns an
    ``AllToAllTokenDispatcher`` (ep>1) whose EP submesh is wired later (after the
    device mesh is built, during ``_parallelize``); until wired it falls back to
    local routing. The cross-rank all-to-all collectives are device-bound (HV-10).
    """
    parallelizer = resolve_moe_parallelizer(dtensor_cfg)
    token_dispatcher = parallelizer["token_dispatcher"]

    if token_dispatcher == "local":
        return LocalTokenDispatcher(
            num_experts=num_experts,
            top_k=top_k,
            score_before_experts=score_before_experts,
        )
    if token_dispatcher == "alltoall":
        return AllToAllTokenDispatcher(
            num_experts=num_experts,
            top_k=top_k,
            score_before_experts=score_before_experts,
        )
    # Unreachable after resolve_moe_parallelizer; defensive.
    raise ValueError(f"unhandled token_dispatcher {token_dispatcher!r}")


def _extract_num_routed_experts(hf_config: Any) -> Optional[int]:
    """Pull the total routed-expert count from an HF config (or its text_config).

    Pure (no I/O); returns ``None`` when no recognized attribute is present.
    """
    for cfg in (hf_config, getattr(hf_config, "text_config", None)):
        if cfg is None:
            continue
        for attr in _ROUTED_EXPERT_COUNT_ATTRS:
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    return None


def read_num_routed_experts(
    model_name: str, hf_config_overrides: Optional[dict[str, Any]] = None
) -> Optional[int]:
    """Best-effort routed-expert count from a model's HF config.

    Deferred ``transformers`` import (config-only read, no model load). Returns
    ``None`` if the architecture exposes no recognized routed-expert attribute.
    """
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hf_config_overrides:
        for key, value in hf_config_overrides.items():
            setattr(hf_config, key, value)
    return _extract_num_routed_experts(hf_config)
