# Grader factory: turn the env.osworld config into the callable a backend's
# evaluate() invokes (signature ``grader(task_config, handle) -> float``).
#
# Resolution order for cfg["grader"]:
#   - a callable        -> used as-is (programmatic injection).
#   - None / False / "" -> None; the backend keeps raising (no grading configured).
#   - any truthy value  -> the registry-backed custom dispatch (evaluate_custom),
#                          after importing cfg["evaluator_modules"] so task-set
#                          evaluators self-register.
# cfg["judge"] (optional callable) is threaded to evaluators that use it
# (screenshot_judge).

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Optional, cast

from .dispatch import evaluate_custom

logger = logging.getLogger("dockyard_rl.cua.grading.factory")

Grader = Callable[[dict[str, Any], Any], float]


def build_grader(cfg: dict[str, Any]) -> Optional[Grader]:
    """Build the grader callable for a backend from its config, or None."""
    grader = cfg.get("grader")
    if callable(grader):
        return cast(Grader, grader)
    if not grader:
        return None

    for module in cfg.get("evaluator_modules", []) or []:
        importlib.import_module(module)

    judge = cast("Optional[Callable[..., float]]", cfg.get("judge"))
    if judge is not None and not callable(judge):
        raise ValueError("env.osworld.judge must be a callable if set")

    def _grader(task_config: dict[str, Any], handle: Any) -> float:
        return evaluate_custom(task_config, handle, judge=judge)

    return _grader
