# Reference custom evaluators, self-registered on import.
#
# These grade from channels every CUA backend exposes — the agent's action
# history (pure), a fixed score (scaffolding / out-of-band grading), and a
# screenshot routed to an injected judge model. They deliberately avoid the
# OSWorld guest server (:5000), which the Windows / macOS guests do not run.
# Author task-set-specific evaluators alongside these with @register_evaluator
# and list their module under env.osworld.evaluator_modules.

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from .registry import register_evaluator

logger = logging.getLogger("dockyard_rl.cua.grading.builtins")


@register_evaluator("manual")
def manual(
    task_config: dict[str, Any],
    handle: Any,
    *,
    options: dict[str, Any],
    judge: Optional[Callable[..., float]] = None,
) -> float:
    """Return a fixed score (default 1.0).

    For tasks graded out-of-band or scaffolded before a real evaluator exists.
    The FAIL gate still applies upstream in dispatch, so a trailing FAIL on a
    non-infeasible task scores 0 regardless of this value.
    """
    return float(options.get("score", 1.0))


@register_evaluator("final_action_match")
def final_action_match(
    task_config: dict[str, Any],
    handle: Any,
    *,
    options: dict[str, Any],
    judge: Optional[Callable[..., float]] = None,
) -> float:
    """Score 1.0 iff the agent's last recorded action matches a predicate.

    options (any subset, all that are present must hold):
        equals:   exact-match the last action string.
        contains: the last action string contains this substring.
        control:  one of "done" / "fail" / "wait" — matches the uppercase token
                  the backends record for control actions.
    """
    history = list(getattr(handle, "action_history", []) or [])
    if not history:
        return 0.0
    last = history[-1]
    last_str = last if isinstance(last, str) else str(last)

    checks: list[bool] = []
    if "equals" in options:
        checks.append(last_str == str(options["equals"]))
    if "contains" in options:
        checks.append(str(options["contains"]) in last_str)
    if "control" in options:
        checks.append(last_str == str(options["control"]).upper())
    if not checks:
        raise ValueError(
            "final_action_match requires at least one of options.equals / "
            "contains / control"
        )
    return 1.0 if all(checks) else 0.0


@register_evaluator("screenshot_judge")
def screenshot_judge(
    task_config: dict[str, Any],
    handle: Any,
    *,
    options: dict[str, Any],
    judge: Optional[Callable[..., float]] = None,
) -> float:
    """Grade the final screen with an injected judge model.

    Captures the current frame via the backend's The-Eye client and passes it to
    ``judge(image_bytes, instruction, **options) -> float`` (configured as
    env.osworld.judge). Without a judge configured this raises, rather than
    silently scoring 0 — screenshot grading needs a model.
    """
    if judge is None:
        raise ValueError(
            "screenshot_judge needs a judge callable; set env.osworld.judge to a "
            "callable(image_bytes, instruction, **options) -> float."
        )
    eye = getattr(handle, "eye", None)
    if eye is None:
        raise ValueError("screenshot_judge requires a backend handle with an 'eye' client")
    try:
        image_bytes = eye.snapshot()
    except Exception as exc:  # noqa: BLE001 - a dropped frame should score 0, not crash grading
        logger.error("screenshot_judge: The-Eye snapshot failed: %s", exc)
        return 0.0
    instruction = task_config.get("instruction", "") or getattr(
        handle, "task_config", {}
    ).get("instruction", "")
    return float(judge(image_bytes, instruction, **options))
