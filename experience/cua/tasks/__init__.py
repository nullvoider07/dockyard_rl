# Custom CUA task authoring.
#
# Custom task sets (Windows / macOS, or extra Linux tasks) reuse the OSWorld
# task_config JSON shape so experience/cua/datasets/osworld.py and
# experience/cua/environment.py load and run them unchanged: point the dataset's
# data_dir at a tree of <split_file> + examples/<domain>/<id>.json. The only
# difference from the OSWorld suite is the evaluator block, which names a
# registered custom evaluator (experience/cua/grading) instead of an OSWorld
# getter/metric pair. ``validate_task_config`` / ``validate_tree`` check a task
# (or a whole tree) before a run so authoring errors surface offline, not as a
# grading crash mid-rollout.

from __future__ import annotations

from .validate import (
    TaskConfigError,
    validate_task_config,
    validate_tree,
    assert_valid,
)

__all__ = [
    "TaskConfigError",
    "validate_task_config",
    "validate_tree",
    "assert_valid",
]
