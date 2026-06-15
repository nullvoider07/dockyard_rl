# Custom-task grading dispatch.
#
# Grades a finished custom CUA episode from its task_config and the live backend
# handle, returning a reward in [0, 1]. Reuses the OSWorld task_config shape so
# datasets/osworld.py and environment.py stay unchanged: the ``evaluator`` block
# carries ``func`` (a registered evaluator name, or a list of names), an optional
# ``conj`` ("and" | "or"), and ``options`` (a dict, or a list aligned to a func
# list).
#
# The action-history FAIL gate mirrors the vendored OSWorld dispatch
# (vendor/desktop_env_eval/dispatch.py): an "infeasible" task is solved iff the
# agent's last action was FAIL, and any normal task with a trailing FAIL scores 0.

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from .registry import resolve_evaluator

logger = logging.getLogger("dockyard_rl.cua.grading.dispatch")


def _is_fail(last_action: Any) -> bool:
    return last_action == "FAIL" or (
        isinstance(last_action, dict) and last_action.get("action_type") == "FAIL"
    )


def _action_history(handle: Any) -> list[Any]:
    return list(getattr(handle, "action_history", []) or [])


def _options_for(evaluator: dict[str, Any], n: Optional[int]) -> Any:
    """Resolve the per-func option dict(s). Single func -> dict; list -> list."""
    opts = evaluator.get("options")
    if n is None:
        return dict(opts) if isinstance(opts, dict) else {}
    if isinstance(opts, list):
        if len(opts) != n:
            raise ValueError(
                f"evaluator.options length {len(opts)} != func length {n}"
            )
        return [dict(o) if isinstance(o, dict) else {} for o in opts]
    # A single dict broadcast across every func in the list.
    return [dict(opts) if isinstance(opts, dict) else {} for _ in range(n)]


def evaluate_custom(
    task_config: dict[str, Any],
    handle: Any,
    *,
    judge: Optional[Callable[..., float]] = None,
) -> float:
    """Grade a finished custom episode; return reward in [0, 1].

    Args:
        task_config: the task's config dict; its ``evaluator`` block drives grading.
        handle: the backend's live episode handle (duck-typed — evaluators read
            ``action_history`` / ``eye`` / ``cc`` / ``host`` / ``task_config`` off it).
        judge: optional screenshot-judge callable threaded to evaluators that use it.
    """
    evaluator = task_config["evaluator"]
    func = evaluator["func"]

    history = _action_history(handle)
    last_action = history[-1] if history else None

    if func == "infeasible":
        return 1.0 if last_action is not None and _is_fail(last_action) else 0.0
    if last_action is not None and _is_fail(last_action):
        return 0.0

    conj = evaluator.get("conj", "and")

    if isinstance(func, list):
        options_list = _options_for(evaluator, len(func))
        scores: list[float] = []
        for name, opts in zip(func, options_list):
            score = float(resolve_evaluator(name)(
                task_config, handle, options=opts, judge=judge
            ))
            if conj == "and" and score == 0.0:
                return 0.0
            if conj == "or" and score == 1.0:
                return 1.0
            scores.append(score)
        if not scores:
            return 0.0
        return sum(scores) / len(scores) if conj == "and" else max(scores)

    options = _options_for(evaluator, None)
    return float(resolve_evaluator(func)(
        task_config, handle, options=options, judge=judge
    ))
