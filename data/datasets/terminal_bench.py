"""Terminal-Bench 2.1 dataset loader for Project Dockyard RL training.

Trainer-side only. Loads the per-task directories provisioned by
``scripts/cluster/terminal_bench_setup.sh`` (from harbor-framework/
terminal-bench-2-1) and presents each task as a ``messages`` + metadata row.

Each task is a multi-turn terminal episode: the agent drives a long-lived
container (executor session mode) and the task is graded on the FINAL container
state by late-injecting the task's ``tests/`` and running ``test.sh`` (pytest →
``/logs/verifier/ctrf.json`` + ``reward.txt``). This loader embeds everything
the environment needs to start the session and run the verifier:

  * the image to run — a prebuilt ``docker_image`` from ``task.toml`` (pull) or,
    when absent, the task's ``environment/Dockerfile`` + build context (build);
  * the held-out ``tests/`` (injected only at finish — never shown to the agent);
  * the verifier command, mount, result-file paths, timeouts, env, and the
    network policy.

The agent prompt is built only from ``instruction.md`` plus a fixed terminal
protocol system prompt; the tests/solution are never in the prompt.

task.toml fields read (schema 1.1):
    [environment] docker_image, build_timeout_sec, allow_internet, env
    [agent] timeout_sec      [verifier] timeout_sec
"""

from __future__ import annotations

import base64
import json
import os
import tomllib
from typing import Any, Optional

from datasets import Dataset
from dockyard_rl.data.datasets.raw_dataset import RawDataset

_SYSTEM_PROMPT = """\
You are an expert operating a Linux terminal to accomplish a task. You work over
multiple turns.

Protocol:
- Each turn, reply with exactly ONE shell command inside a single ```bash code
  fence, and nothing else. I will run it and reply with its combined
  stdout/stderr.
- When the task is fully complete, reply with exactly the line TASK_COMPLETE and
  no code fence. I will then run the hidden verifier.
- Each command runs in a fresh non-interactive shell at the image's working
  directory; a `cd` or shell variable does not persist to the next turn. Chain
  steps within one command (`cd dir && ...`) or use absolute paths.
- The task is graded on the final state of the filesystem, not on your console
  output, so make the required changes durable.\
"""


def _read_payload(path: str) -> Any:
    """Read a file as UTF-8 text, or a ``{"b64": ...}`` wrapper if binary."""
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"b64": base64.b64encode(raw).decode("ascii")}


def _load_dir(root: str, exclude: Optional[set[str]] = None) -> dict[str, Any]:
    """Map every file under ``root`` to its payload, keyed by relative path."""
    exclude = exclude or set()
    out: dict[str, Any] = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            if rel in exclude:
                continue
            out[rel] = _read_payload(full)
    return out


def _runnable_mode(task_dir: str, toml_env: dict) -> Optional[str]:
    """Return "image" (prebuilt ref), "build" (Dockerfile), or None (skip)."""
    if (toml_env.get("docker_image") or "").strip():
        return "image"
    if os.path.isfile(os.path.join(task_dir, "environment", "Dockerfile")):
        return "build"
    return None


class TerminalBenchDataset(RawDataset):
    """Terminal-Bench 2.1 tasks exposed as ``messages`` + metadata rows.

    Args:
        tasks_dir:    Directory of per-task dirs produced by
                      ``terminal_bench_setup.sh``. Defaults to the
                      ``DOCKYARD_TERMINAL_BENCH_TASKS_DIR`` env var.
        task_ids:     Optional subset of task ids (directory names).
        max_turns:    Per-episode agent turn budget embedded into each row; the
                      environment self-terminates with the verifier score on the
                      last allowed turn. Keep ``grpo.max_rollout_turns`` above it.
        exec_timeout_sec: Per-command wall-clock cap for the agent's commands.
        shuffle_seed: If set, shuffle with this seed.
    """

    def __init__(
        self,
        tasks_dir: Optional[str] = None,
        task_ids: Optional[list[str]] = None,
        max_turns: int = 30,
        exec_timeout_sec: int = 120,
        shuffle_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.task_name = "terminal_bench"
        self.max_turns = int(max_turns)
        self.exec_timeout_sec = int(exec_timeout_sec)
        self.tasks_dir = tasks_dir or os.environ.get(
            "DOCKYARD_TERMINAL_BENCH_TASKS_DIR", ""
        )

        if not self.tasks_dir or not os.path.isdir(self.tasks_dir):
            raise ValueError(
                "Terminal-Bench requires the provisioned task tree. Set tasks_dir "
                "(or DOCKYARD_TERMINAL_BENCH_TASKS_DIR) to the directory populated "
                "by scripts/cluster/terminal_bench_setup.sh; "
                f"got {self.tasks_dir!r}."
            )

        names = sorted(
            d for d in os.listdir(self.tasks_dir)
            if os.path.isfile(os.path.join(self.tasks_dir, d, "task.toml"))
        )
        if task_ids:
            wanted = set(task_ids)
            names = [n for n in names if n in wanted]

        print(
            f"Loading Terminal-Bench tasks from {self.tasks_dir} ({len(names)} found)",
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
                f"  ⚠ Skipped {skipped} task(s) with no prebuilt image and no "
                "environment/Dockerfile (compose-only or malformed)",
                flush=True,
            )
        if not rows:
            raise ValueError(f"No runnable Terminal-Bench tasks under {self.tasks_dir}")

        self.dataset = Dataset.from_list(rows)
        if shuffle_seed is not None:
            self.dataset = self.dataset.shuffle(seed=shuffle_seed)
        self.val_dataset = None
        print(f"  ✓ Loaded {len(self.dataset)} Terminal-Bench tasks", flush=True)

    def _build_row(self, task_id: str) -> Optional[dict[str, Any]]:
        task_dir = os.path.join(self.tasks_dir, task_id)
        with open(os.path.join(task_dir, "task.toml"), "rb") as fh:
            toml = tomllib.load(fh)

        env_cfg  = toml.get("environment", {}) or {}
        agent    = toml.get("agent", {}) or {}
        verifier = toml.get("verifier", {}) or {}

        mode = _runnable_mode(task_dir, env_cfg)
        if mode is None:
            return None

        instruction_path = os.path.join(task_dir, "instruction.md")
        with open(instruction_path, encoding="utf-8") as fh:
            instruction = fh.read().strip()

        # Held-out verifier: every file under tests/ (injected only at finish).
        tests = _load_dir(os.path.join(task_dir, "tests"))
        if not tests:
            return None

        image = ""
        dockerfile = ""
        build_context: dict[str, Any] = {}
        if mode == "image":
            image = str(env_cfg["docker_image"]).strip()
        else:
            env_dir = os.path.join(task_dir, "environment")
            with open(os.path.join(env_dir, "Dockerfile"), encoding="utf-8") as fh:
                dockerfile = fh.read()
            build_context = _load_dir(env_dir, exclude={"Dockerfile"})

        container_env = env_cfg.get("env", {}) or {}

        return {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": instruction},
            ],
            "task_name":           self.task_name,
            "task_id":             task_id,
            "mode":                mode,
            "image":               image,
            "dockerfile":          dockerfile,
            "build_context_json":  json.dumps(build_context),
            "tests_json":          json.dumps(tests),
            "container_env_json":  json.dumps(container_env),
            "allow_internet":      bool(env_cfg.get("allow_internet", True)),
            "harness_mount":       "/tests",
            # Precreate the log dir the verifier writes into, then run test.sh.
            "test_command":        "mkdir -p /logs/verifier && bash /tests/test.sh",
            "result_files":        ["/logs/verifier/ctrf.json", "/logs/verifier/reward.txt"],
            "agent_timeout_sec":   int(agent.get("timeout_sec", 900)),
            "verifier_timeout_sec": int(verifier.get("timeout_sec", 900)),
            "build_timeout_sec":   int(env_cfg.get("build_timeout_sec", 1800)),
            "exec_timeout_sec":    self.exec_timeout_sec,
            "max_turns":           self.max_turns,
        }
