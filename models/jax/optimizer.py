"""optax optimizer + LR schedule construction from the torch-style config.

Mirrors the config surface used by the torch worker (`policy.optimizer` =
AdamW kwargs; `policy.scheduler` = a list of torch scheduler specs;
`policy.max_grad_norm`). J4 supports AdamW + a LinearLR-warmup→Constant
schedule (the shipped GRPO config); unrecognized schedulers fall back to a
constant LR. The returned schedule callable is kept so the worker can log the
current LR.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import optax

# optax's own schedule type (callable step -> lr); aliased for readability and to
# match what optax.{constant,linear,join}_schedule return and adamw accepts.
Schedule = optax.Schedule


def build_schedule(base_lr: float, scheduler: Optional[Sequence[Mapping[str, Any]]]) -> Schedule:
    """Build an LR schedule from the torch scheduler-spec list.

    Recognizes ``LinearLR`` (warmup: ``start_factor`` -> ``end_factor`` over
    ``total_iters``) optionally followed by ``ConstantLR``. Anything else yields
    a constant ``base_lr``.
    """
    if not scheduler:
        return optax.constant_schedule(base_lr)

    first = scheduler[0]
    name = str(first.get("name", "")).split(".")[-1]
    kw = first.get("kwargs", {}) or {}
    if name == "LinearLR":
        start_factor = float(kw.get("start_factor", 1.0 / 3.0))
        end_factor = float(kw.get("end_factor", 1.0))
        total_iters = int(kw.get("total_iters", 0))
        if total_iters <= 0:
            return optax.constant_schedule(base_lr)
        warmup = optax.linear_schedule(
            init_value=base_lr * start_factor,
            end_value=base_lr * end_factor,
            transition_steps=total_iters,
        )
        # After warmup, hold at base_lr*end_factor (ConstantLR or implicit).
        constant = optax.constant_schedule(base_lr * end_factor)
        return optax.join_schedules([warmup, constant], [total_iters])

    return optax.constant_schedule(base_lr)


def build_optimizer(
    optimizer_cfg: Mapping[str, Any],
    scheduler_cfg: Optional[Sequence[Mapping[str, Any]]] = None,
    max_grad_norm: Optional[float] = None,
) -> tuple[optax.GradientTransformation, Schedule]:
    """Build ``(tx, schedule)`` from the torch-style optimizer/scheduler config.

    ``optimizer_cfg`` follows ``{"name": "...AdamW", "kwargs": {lr, weight_decay,
    betas, eps, ...}}``. Gradient clipping (``optax.clip_by_global_norm``) is
    prepended when ``max_grad_norm`` is a positive finite value.
    """
    name = str(optimizer_cfg.get("name", "AdamW")).split(".")[-1]
    kw = dict(optimizer_cfg.get("kwargs", {}) or {})
    base_lr = float(kw.get("lr", 1e-3))
    betas = kw.get("betas", (0.9, 0.999))
    b1, b2 = float(betas[0]), float(betas[1])
    eps = float(kw.get("eps", 1e-8))
    weight_decay = float(kw.get("weight_decay", 0.0))

    schedule = build_schedule(base_lr, scheduler_cfg)

    if name == "AdamW":
        core = optax.adamw(learning_rate=schedule, b1=b1, b2=b2, eps=eps, weight_decay=weight_decay)
    elif name == "Adam":
        core = optax.adam(learning_rate=schedule, b1=b1, b2=b2, eps=eps)
    else:
        raise NotImplementedError(f"Optimizer {name!r} not supported in J4 (AdamW/Adam only).")

    transforms: list[optax.GradientTransformation] = []
    if max_grad_norm is not None and max_grad_norm > 0:
        transforms.append(optax.clip_by_global_norm(max_grad_norm))
    transforms.append(core)
    return optax.chain(*transforms), schedule
