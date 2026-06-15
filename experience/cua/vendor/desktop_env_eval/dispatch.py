# Standalone OSWorld grading dispatch, decoupled from DesktopEnv.
#
# Replicates desktop_env.desktop_env.DesktopEnv._set_task_info + .evaluate()
# (pinned 705623ca18e0055dd995fd5a350d6588cff2caf5): resolve the task's
# evaluator spec to metric + getter callables, pull result/expected state via
# the getters, and score in [0, 1] with the configured conjunction. The single
# behavioural change vs upstream is the FileNotFoundError handling in the
# multi-metric "or" branch — upstream leaves result_state unbound there; here a
# missing file scores that metric 0 instead of raising.

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from . import getters, metrics
from .grading_env import GradingEnv

logger = logging.getLogger("dockyard_rl.cua.vendor.desktop_env_eval.dispatch")


def _is_fail(last_action: Any) -> bool:
    return last_action == "FAIL" or (
        isinstance(last_action, dict) and last_action.get("action_type") == "FAIL"
    )


def _resolve_metric(func: Any) -> Any:
    return (
        [getattr(metrics, f) for f in func]
        if isinstance(func, list)
        else getattr(metrics, func)
    )


def _resolve_getter(spec: Any) -> Any:
    # spec is the evaluator "result"/"expected" config (dict, list of dict, or
    # absent). Resolve get_<type> on the getters package, mirroring upstream.
    if isinstance(spec, list):
        return [
            getattr(getters, "get_{:}".format(s["type"])) if s else None
            for s in spec
        ]
    return getattr(getters, "get_{:}".format(spec["type"]))


def evaluate(
    task_config: dict[str, Any],
    env: GradingEnv,
    setup_fn: Optional[Callable[[list[dict[str, Any]]], None]] = None,
) -> float:
    """Grade a finished OSWorld episode; return reward in [0, 1].

    Args:
        task_config: the OSWorld task_config dict (its ``evaluator`` block drives
            grading).
        env: GradingEnv pointed at the host running the guest server.
        setup_fn: optional runner for the evaluator's ``postconfig`` setup steps
            (the backend supplies its SetupController-backed runner). When None,
            a non-empty postconfig is skipped with a warning.
    """
    evaluator = task_config["evaluator"]

    postconfig = evaluator.get("postconfig", [])
    if postconfig:
        if setup_fn is not None:
            setup_fn(postconfig)
        else:
            logger.warning(
                "evaluator.postconfig has %d step(s) but no setup_fn was "
                "provided; skipping postconfig (grading may be inaccurate)",
                len(postconfig),
            )

    func = evaluator["func"]

    # Action-history gate: an "infeasible" task is solved iff the agent's last
    # action was FAIL; a normal task with a trailing FAIL scores 0.
    last_action = env.action_history[-1] if env.action_history else None
    if func == "infeasible":
        return 1.0 if last_action is not None and _is_fail(last_action) else 0.0
    if last_action is not None and _is_fail(last_action):
        return 0.0

    metric: Any = _resolve_metric(func)
    conj = evaluator.get("conj", "and")

    has_result = "result" in evaluator and len(evaluator["result"]) > 0
    result_getter: Any = _resolve_getter(evaluator["result"]) if has_result else (
        [None] * len(metric) if isinstance(metric, list) else None
    )

    has_expected = "expected" in evaluator and len(evaluator["expected"]) > 0
    expected_getter: Any = _resolve_getter(evaluator["expected"]) if has_expected else (
        [None] * len(metric) if isinstance(metric, list) else None
    )

    if "options" in evaluator:
        opts = evaluator["options"]
        metric_options: Any = (
            [o if o else {} for o in opts] if isinstance(opts, list) else opts
        )
    else:
        metric_options = [{}] * len(metric) if isinstance(metric, list) else {}

    if isinstance(metric, list):
        assert len(metric) == len(result_getter), (
            "metric/result_getter length mismatch"
        )
        if has_expected:
            assert len(metric) == len(expected_getter), (
                "metric/expected_getter length mismatch"
            )
        results: list[float] = []
        for idx, m in enumerate(metric):
            config = evaluator["result"][idx]
            try:
                result_state = result_getter[idx](env, config)
            except FileNotFoundError:
                logger.error("File not found grading metric %d", idx)
                if conj == "and":
                    return 0.0
                results.append(0.0)
                continue

            if has_expected and evaluator["expected"]:
                expected_state = expected_getter[idx](env, evaluator["expected"][idx])
                score = float(m(result_state, expected_state, **metric_options[idx]))
            else:
                score = float(m(result_state, **metric_options[idx]))

            if conj == "and" and score == 0.0:
                return 0.0
            if conj == "or" and score == 1.0:
                return 1.0
            results.append(score)
        return sum(results) / len(results) if conj == "and" else max(results)

    # Single metric.
    try:
        result_state = result_getter(env, evaluator["result"])
    except FileNotFoundError:
        logger.error("File not found grading single metric")
        return 0.0

    if has_expected and evaluator["expected"]:
        expected_state = expected_getter(env, evaluator["expected"])
        return float(metric(result_state, expected_state, **metric_options))
    return float(metric(result_state, **metric_options))
