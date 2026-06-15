"""ProgramBench dataset loader for Project Dockyard RL training.

Trainer-side only. Loads the merged task tree provisioned by
``scripts/cluster/program_bench_setup.sh`` (task.yaml + tests.json from
facebookresearch/ProgramBench, tests/<branch>.tar.gz from HF
programbench/ProgramBench-Tests) and presents each instance as a ``messages`` +
metadata row.

Each task is a multi-turn cleanroom-reconstruction episode: the agent probes an
execute-only reference binary (``/workspace/executable``) and documentation that
ship inside the per-instance image ``programbench/<id>:task_cleanroom``, then
writes source + a ``compile.sh`` that rebuilds ``./executable`` in the task's
fixed source language. Grading (in the reward) re-extracts the submission into a
clean image, runs ``compile.sh``, and runs the held-out behavioural pytest
"branches" against the rebuilt executable, scoring passed node IDs against
``tests.json``. This loader embeds, per instance, the image ref and the selected
branches' test tars (base64) + expected node IDs; the agent never sees them.

task.yaml fields: repository, commit, language, difficulty, eval_clean_hashes.
tests.json: ``{"branches": {<hash>: {"ignored": bool, "tests": [node_id, ...]}}}``.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Optional

import yaml
from datasets import Dataset
from dockyard_rl.data.datasets.raw_dataset import RawDataset

_SYSTEM_PROMPT = """\
You are an expert software engineer doing cleanroom reconstruction in a Linux
terminal, over multiple turns, with no internet access.

In your workspace (/workspace) there is an execute-only reference binary at
./executable and its documentation. Your job: reconstruct a program, in the
required source language, that is behaviourally identical to ./executable.

Protocol:
- Each turn, reply with exactly ONE shell command inside a single ```bash code
  fence, and nothing else. I will run it and reply with its combined
  stdout/stderr. Run ./executable with inputs/flags to observe its behaviour;
  read the documentation; then write your source files.
- You may NOT read or decompile the binary's contents; observe it only by
  running it (black-box).
- Produce a ./compile.sh that builds your sources into ./executable from a clean
  checkout (it must install/compile with only what the image already provides —
  there is no network).
- When done, reply with exactly the line TASK_COMPLETE (no code fence). Grading
  then rebuilds your submission from scratch and runs hidden behavioural tests
  against the produced ./executable.
- Each command runs in a fresh non-interactive shell at /workspace; a `cd` or
  shell variable does not persist to the next turn — chain steps in one command
  or use absolute paths.\
"""

_USER_TEMPLATE = """\
Reconstruct the program behaviourally, implemented in {language}.

The reference binary is at /workspace/executable (execute-only) and its
documentation is in /workspace. Probe the binary, then write your {language}
sources and a /workspace/compile.sh that builds them into /workspace/executable.
Reply TASK_COMPLETE when your build is ready for grading.\
"""


def _select_branches(
    tests_json: dict, tests_dir: str, max_branches: Optional[int]
) -> dict[str, dict]:
    """Pick up to max_branches non-ignored branches that have a tar on disk.

    Returns {branch: {"tar_b64": str, "expected": [node_id, ...]}}. max_branches
    None or <= 0 selects all eligible branches (sorted by name for determinism).
    """
    branches = tests_json.get("branches") or {}
    selected: dict[str, dict] = {}
    for name in sorted(branches):
        spec = branches[name] or {}
        if spec.get("ignored"):
            continue
        tar_path = os.path.join(tests_dir, f"{name}.tar.gz")
        if not os.path.isfile(tar_path):
            continue
        expected = [str(t) for t in (spec.get("tests") or [])]
        if not expected:
            continue
        with open(tar_path, "rb") as fh:
            tar_b64 = base64.b64encode(fh.read()).decode("ascii")
        selected[name] = {"tar_b64": tar_b64, "expected": expected}
        if max_branches and max_branches > 0 and len(selected) >= max_branches:
            break
    return selected


class ProgramBenchDataset(RawDataset):
    """ProgramBench cleanroom-reconstruction tasks as ``messages`` + metadata rows.

    Args:
        tasks_dir:     Provisioned merged tree (program_bench_setup.sh output).
                       Defaults to the DOCKYARD_PROGRAM_BENCH_DIR env var.
        image_registry/image_tag: Compose the per-instance image as
                       ``{registry}/{id with "__"→"_1776_"}:{tag}`` (the official
                       programbench/...:task_cleanroom naming).
        max_branches:  Max held-out test branches embedded/graded per task
                       (None/0 = all eligible). Bounds dataset size and grading
                       cost; the reward scores passed/total over their union.
        task_ids:      Optional subset of instance ids.
        max_turns / exec_timeout_sec / grading_timeout_sec: per-episode budgets.
        shuffle_seed:  If set, shuffle with this seed.
    """

    def __init__(
        self,
        tasks_dir: Optional[str] = None,
        image_registry: str = "programbench",
        image_tag: str = "task_cleanroom",
        max_branches: Optional[int] = 2,
        task_ids: Optional[list[str]] = None,
        max_turns: int = 40,
        exec_timeout_sec: int = 120,
        grading_timeout_sec: int = 3600,
        shuffle_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.task_name = "program_bench"
        self.image_registry = image_registry.rstrip("/")
        self.image_tag = image_tag
        self.max_branches = max_branches
        self.max_turns = int(max_turns)
        self.exec_timeout_sec = int(exec_timeout_sec)
        self.grading_timeout_sec = int(grading_timeout_sec)
        self.tasks_dir = tasks_dir or os.environ.get("DOCKYARD_PROGRAM_BENCH_DIR", "")

        if not self.tasks_dir or not os.path.isdir(self.tasks_dir):
            raise ValueError(
                "ProgramBench requires the provisioned task tree. Set tasks_dir "
                "(or DOCKYARD_PROGRAM_BENCH_DIR) to the directory populated by "
                "scripts/cluster/program_bench_setup.sh; "
                f"got {self.tasks_dir!r}."
            )

        names = sorted(
            d for d in os.listdir(self.tasks_dir)
            if os.path.isfile(os.path.join(self.tasks_dir, d, "task.yaml"))
            and os.path.isfile(os.path.join(self.tasks_dir, d, "tests.json"))
        )
        if task_ids:
            wanted = set(task_ids)
            names = [n for n in names if n in wanted]

        print(
            f"Loading ProgramBench tasks from {self.tasks_dir} ({len(names)} found)",
            flush=True,
        )

        rows: list[dict[str, Any]] = []
        skipped = 0
        for name in names:
            row = self._build_row(name)
            if row is None:
                skipped += 1
                continue
            rows.append(row)
        if skipped:
            print(
                f"  ⚠ Skipped {skipped} task(s) with no eligible test branch on disk",
                flush=True,
            )
        if not rows:
            raise ValueError(f"No runnable ProgramBench tasks under {self.tasks_dir}")

        self.dataset = Dataset.from_list(rows)
        if shuffle_seed is not None:
            self.dataset = self.dataset.shuffle(seed=shuffle_seed)
        self.val_dataset = None
        print(f"  ✓ Loaded {len(self.dataset)} ProgramBench tasks", flush=True)

    def _image_for(self, instance_id: str) -> str:
        slug = instance_id.replace("__", "_1776_")
        return f"{self.image_registry}/{slug}:{self.image_tag}"

    def _build_row(self, instance_id: str) -> Optional[dict[str, Any]]:
        task_dir = os.path.join(self.tasks_dir, instance_id)
        with open(os.path.join(task_dir, "task.yaml"), encoding="utf-8") as fh:
            task = yaml.safe_load(fh) or {}
        with open(os.path.join(task_dir, "tests.json"), encoding="utf-8") as fh:
            tests_json = json.load(fh)

        language = (task.get("language") or "").strip() or "unknown"
        branches = _select_branches(
            tests_json, os.path.join(task_dir, "tests"), self.max_branches
        )
        if not branches:
            return None

        user_content = _USER_TEMPLATE.format(language=language)
        return {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            "task_name":           self.task_name,
            "task_id":             instance_id,
            "language":            language,
            "image":               self._image_for(instance_id),
            "branches_json":       json.dumps(branches),
            "max_turns":           self.max_turns,
            "exec_timeout_sec":    self.exec_timeout_sec,
            "grading_timeout_sec": self.grading_timeout_sec,
        }
