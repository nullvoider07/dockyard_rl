# Custom CUA evaluator registry.
#
# OSWorld's task suite + evaluators are Ubuntu-authored and grade through the
# guest server (:5000), which the Windows / macOS guests do not run. Custom task
# sets for those platforms therefore reference a named evaluator here instead of
# an OSWorld getter/metric pair. A registered evaluator grades the finished
# episode directly from the live backend handle and returns a reward in [0, 1].
#
# Mirrors the data/processors.py registry pattern (dict + register decorator,
# raise on duplicate / unknown name).

from __future__ import annotations

from typing import Callable, Optional

# A custom evaluator is called as
#   fn(task_config, handle, *, options=<dict>, judge=<callable|None>) -> float
# where ``handle`` is the backend's live episode handle (duck-typed: evaluators
# read whatever they need — ``action_history``, ``eye``, ``cc``, ``host``,
# ``task_config`` — off it), ``options`` is the evaluator block's per-func option
# dict, and ``judge`` is the optional screenshot-judge callable threaded from the
# grader config.
EvaluatorFn = Callable[..., float]

EVALUATOR_REGISTRY: dict[str, EvaluatorFn] = {}


def register_evaluator(name: str) -> Callable[[EvaluatorFn], EvaluatorFn]:
    """Register a custom evaluator under ``name``. Raises on duplicate name."""

    def _decorator(fn: EvaluatorFn) -> EvaluatorFn:
        if name in EVALUATOR_REGISTRY:
            raise ValueError(f"Custom evaluator {name!r} is already registered")
        EVALUATOR_REGISTRY[name] = fn
        return fn

    return _decorator


def resolve_evaluator(name: str) -> EvaluatorFn:
    """Look up a registered evaluator by name; raise with the available set."""
    fn: Optional[EvaluatorFn] = EVALUATOR_REGISTRY.get(name)
    if fn is None:
        available = ", ".join(sorted(EVALUATOR_REGISTRY)) or "<none>"
        raise ValueError(
            f"Unknown custom evaluator {name!r}. Registered evaluators: {available}. "
            "Register one with @register_evaluator or list its module under "
            "env.osworld.evaluator_modules."
        )
    return fn
