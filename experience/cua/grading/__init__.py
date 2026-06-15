# Custom CUA task grading: a named-evaluator registry + dispatch for task sets
# OSWorld does not cover (custom Windows / macOS tasks). Importing the package
# self-registers the reference evaluators (.builtins).

from __future__ import annotations

from .registry import (
    EVALUATOR_REGISTRY,
    EvaluatorFn,
    register_evaluator,
    resolve_evaluator,
)
from .dispatch import evaluate_custom
from .factory import build_grader, Grader

from . import builtins as _builtins  # noqa: F401 - import for self-registration side effect

__all__ = [
    "EVALUATOR_REGISTRY",
    "EvaluatorFn",
    "register_evaluator",
    "resolve_evaluator",
    "evaluate_custom",
    "build_grader",
    "Grader",
]
