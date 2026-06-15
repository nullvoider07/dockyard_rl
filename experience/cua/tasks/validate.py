# Validate a custom CUA task_config (or a whole task tree) before a run.
#
# Catches the authoring mistakes that would otherwise only surface as a grading
# crash mid-rollout: a missing instruction/evaluator, an evaluator.func that no
# registered evaluator (experience/cua/grading) provides, a bad conj, or an
# options list that does not line up with a func list. Pure offline checks — no
# guest, no model.

from __future__ import annotations

import importlib
import json
import os
from typing import Any, Iterable, Optional

from dockyard_rl.experience.cua.grading import EVALUATOR_REGISTRY

_VALID_CONJ = {"and", "or"}
_REQUIRED_FIELDS = ("id", "instruction", "evaluator")
# Grading dispatch handles "infeasible" via the action-history FAIL gate, so it
# is always valid even though no evaluator function is registered under it.
_GATE_FUNCS = {"infeasible"}


class TaskConfigError(ValueError):
    """Raised by ``assert_valid`` when a task_config fails validation."""


def _import_modules(modules: Optional[Iterable[str]]) -> None:
    for module in modules or []:
        importlib.import_module(module)


def validate_task_config(
    task_config: Any,
    *,
    require_registered: bool = True,
) -> list[str]:
    """Return a list of problems with ``task_config`` (empty == valid).

    Args:
        task_config: the task config dict to check.
        require_registered: when True, every evaluator.func name (other than the
            FAIL-gate ``infeasible``) must be present in the custom evaluator
            registry. Set False to validate shape only (e.g. before the task
            set's evaluator modules are importable).
    """
    problems: list[str] = []

    if not isinstance(task_config, dict):
        return ["task_config must be a JSON object"]

    for field in _REQUIRED_FIELDS:
        if field not in task_config:
            problems.append(f"missing required field {field!r}")

    instruction = task_config.get("instruction")
    if instruction is not None and not str(instruction).strip():
        problems.append("instruction is empty")

    evaluator = task_config.get("evaluator")
    if not isinstance(evaluator, dict):
        if "evaluator" in task_config:
            problems.append("evaluator must be a JSON object")
        return problems

    func = evaluator.get("func")
    if func is None:
        problems.append("evaluator.func is required")
    names = func if isinstance(func, list) else [func]
    if isinstance(func, list) and not func:
        problems.append("evaluator.func list is empty")

    for name in names:
        if name is None:
            continue
        if not isinstance(name, str):
            problems.append(f"evaluator.func entries must be strings; got {name!r}")
            continue
        if name in _GATE_FUNCS:
            continue
        if require_registered and name not in EVALUATOR_REGISTRY:
            available = ", ".join(sorted(EVALUATOR_REGISTRY)) or "<none>"
            problems.append(
                f"evaluator.func {name!r} is not a registered evaluator "
                f"(available: {available}; list its module under "
                "env.osworld.evaluator_modules)"
            )

    conj = evaluator.get("conj", "and")
    if conj not in _VALID_CONJ:
        problems.append(f"evaluator.conj must be one of {sorted(_VALID_CONJ)}; got {conj!r}")

    options = evaluator.get("options")
    if isinstance(func, list) and isinstance(options, list) and len(options) != len(func):
        problems.append(
            f"evaluator.options length {len(options)} != evaluator.func length {len(func)}"
        )

    config = task_config.get("config", [])
    if config and not isinstance(config, list):
        problems.append("config must be a list of setup steps")
    elif isinstance(config, list):
        for i, step in enumerate(config):
            if not isinstance(step, dict) or "type" not in step:
                problems.append(f"config[{i}] must be a JSON object with a 'type'")

    return problems


def assert_valid(task_config: dict[str, Any], *, require_registered: bool = True) -> None:
    """Raise ``TaskConfigError`` if ``task_config`` is invalid."""
    problems = validate_task_config(task_config, require_registered=require_registered)
    if problems:
        ref = task_config.get("id", "<no id>") if isinstance(task_config, dict) else "<?>"
        raise TaskConfigError(
            f"task {ref!r} failed validation:\n  - " + "\n  - ".join(problems)
        )


def validate_tree(
    data_dir: str,
    *,
    split_file: str = "test_all.json",
    require_registered: bool = True,
    evaluator_modules: Optional[Iterable[str]] = None,
) -> dict[str, list[str]]:
    """Validate every task in an OSWorld-shaped tree; return only the failures.

    Loads ``<data_dir>/<split_file>`` ({domain: [task_id]}) and each
    ``<data_dir>/examples/<domain>/<id>.json``. The returned mapping is keyed by
    ``"<domain>/<task_id>"`` and contains the problem list for each task that did
    not validate (an empty mapping means the whole tree is valid). Imports
    ``evaluator_modules`` first so custom evaluators are registered.
    """
    _import_modules(evaluator_modules)

    split_path = os.path.join(data_dir, split_file)
    with open(split_path, encoding="utf-8") as fh:
        split_index: dict[str, list[str]] = json.load(fh)

    failures: dict[str, list[str]] = {}
    for domain in sorted(split_index):
        for task_id in split_index[domain]:
            ref = f"{domain}/{task_id}"
            cfg_path = os.path.join(data_dir, "examples", domain, f"{task_id}.json")
            if not os.path.isfile(cfg_path):
                failures[ref] = [f"missing task file {cfg_path}"]
                continue
            try:
                with open(cfg_path, encoding="utf-8") as fh:
                    task_config = json.load(fh)
            except json.JSONDecodeError as exc:
                failures[ref] = [f"invalid JSON: {exc}"]
                continue
            problems = validate_task_config(
                task_config, require_registered=require_registered
            )
            if problems:
                failures[ref] = problems
    return failures
