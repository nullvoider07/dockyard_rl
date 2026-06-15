"""OSWorld task-config dataset loader for the CUA experience path.

Trainer-side only. Reads an OSWorld ``evaluation_examples`` tree:

  * ``<data_dir>/<split_file>`` (default ``test_all.json``) maps a domain name to
    a list of task ids, ``{domain: [task_id, ...]}``;
  * each task's full config lives at
    ``<data_dir>/examples/<domain>/<task_id>.json`` (id, instruction, setup
    ``config`` steps, ``related_apps``, ``evaluator``, ``snapshot``, ``proxy``).

Each row presents one task as a text prompt (system protocol + instruction) plus
the full task_config carried as a JSON string in ``extra_env_info``. The desktop
screenshots are NOT in the prompt — the VM is not booted at data-load time; the
CUA rollout (``experience/cua/rollout.py``) injects the per-turn screenshots into
the message log at rollout time. The environment/backend consume the embedded
task_config to provision, drive, and grade the episode.

The data tree is a checkout of xlang-ai/OSWorld's ``evaluation_examples``
(pinned upstream commit 705623ca18e0055dd995fd5a350d6588cff2caf5); point
``data_dir`` (or ``DOCKYARD_OSWORLD_DATA_DIR``) at it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from datasets import Dataset

from dockyard_rl.data.datasets.raw_dataset import RawDataset
from dockyard_rl.data.interfaces import DatumSpec, TaskDataSpec
from dockyard_rl.data.llm_message_utils import get_formatted_message_log
from dockyard_rl.data.processors import PROCESSOR_REGISTRY, register_processor

_SYSTEM_PROMPT = """\
You are an expert agent operating a real {os_label} desktop to accomplish a task.
You work over multiple turns and act through screenshots.

Each turn you receive the current screenshot of the screen (and, when available,
an accessibility tree). Respond with EITHER:
- exactly ONE fenced ```python code block containing pyautogui calls to perform
  the next action(s) (e.g. pyautogui.click(x, y), pyautogui.typewrite("text"),
  pyautogui.hotkey("ctrl", "s")). Use absolute screen coordinates in pixels; the
  screen is {screen_width}x{screen_height}. Keep each turn to a small, verifiable
  step so you can react to the resulting screenshot.
- or exactly one bare control token on its own line:
  - WAIT  — do nothing this turn and wait for the screen to settle;
  - DONE  — the task is fully complete;
  - FAIL  — the task is infeasible and cannot be completed.

Do not wrap a control token in a code fence. The task is graded on the final
state of the machine, not on your console output, so make the required changes
durable before signalling DONE.\
"""


def _build_system_prompt(
    screen_width: int, screen_height: int, os_label: str = "Ubuntu"
) -> str:
    return _SYSTEM_PROMPT.format(
        screen_width=screen_width, screen_height=screen_height, os_label=os_label
    )


class OSWorldDataset(RawDataset):
    """OSWorld tasks exposed as ``messages`` + task_config rows for the CUA path.

    Args:
        data_dir:      OSWorld ``evaluation_examples`` directory. Defaults to the
                       ``DOCKYARD_OSWORLD_DATA_DIR`` env var.
        split_file:    Split index file under ``data_dir`` mapping domain ->
                       [task_id]. Default ``test_all.json``.
        domains:       Optional subset of domain names to include.
        task_ids:      Optional subset of task ids to include.
        include_infeasible: Include tasks whose evaluator is ``infeasible``
                       (solved by emitting FAIL). Default True.
        max_turns:     Per-episode turn budget embedded in each row; the
                       environment self-terminates with the grade on the last
                       allowed turn. Keep ``grpo.max_rollout_turns`` above it.
        screen_width / screen_height: Desktop resolution the agent is told about
                       and the backend renders at. Default 1920x1080.
        pause_after_action: Seconds the backend waits after each action before
                       capturing the next screenshot (OSWorld ``step`` pause).
        os_label:      OS name the agent is told it is operating in the system
                       prompt. Default ``"Ubuntu"`` (the OSWorld suite); set to
                       ``"Windows"`` / ``"macOS"`` for custom task trees on those
                       guests. The action grammar (pyautogui) is unchanged — the
                       backend transpiles it per platform.
        shuffle_seed:  If set, shuffle with this seed.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split_file: str = "test_all.json",
        domains: Optional[list[str]] = None,
        task_ids: Optional[list[str]] = None,
        include_infeasible: bool = True,
        max_turns: int = 15,
        screen_width: int = 1920,
        screen_height: int = 1080,
        pause_after_action: float = 2.0,
        os_label: str = "Ubuntu",
        shuffle_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.task_name = "osworld"
        self.max_turns = int(max_turns)
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self.pause_after_action = float(pause_after_action)
        self.os_label = str(os_label)
        self.data_dir = data_dir or os.environ.get("DOCKYARD_OSWORLD_DATA_DIR", "")

        if not self.data_dir or not os.path.isdir(self.data_dir):
            raise ValueError(
                "OSWorld requires a checkout of the evaluation_examples tree. Set "
                "data_dir (or DOCKYARD_OSWORLD_DATA_DIR) to the directory holding "
                f"{split_file} and examples/; got {self.data_dir!r}."
            )

        split_path = os.path.join(self.data_dir, split_file)
        if not os.path.isfile(split_path):
            raise ValueError(f"OSWorld split file not found: {split_path}")
        with open(split_path, encoding="utf-8") as fh:
            split_index: dict[str, list[str]] = json.load(fh)

        wanted_domains = set(domains) if domains else None
        wanted_ids = set(task_ids) if task_ids else None

        system_prompt = _build_system_prompt(
            self.screen_width, self.screen_height, self.os_label
        )

        rows: list[dict[str, Any]] = []
        skipped_missing = 0
        skipped_infeasible = 0
        for domain in sorted(split_index):
            if wanted_domains is not None and domain not in wanted_domains:
                continue
            for task_id in split_index[domain]:
                if wanted_ids is not None and task_id not in wanted_ids:
                    continue
                cfg_path = os.path.join(
                    self.data_dir, "examples", domain, f"{task_id}.json"
                )
                if not os.path.isfile(cfg_path):
                    skipped_missing += 1
                    continue
                with open(cfg_path, encoding="utf-8") as fh:
                    task_config = json.load(fh)

                evaluator = task_config.get("evaluator", {}) or {}
                if not include_infeasible and evaluator.get("func") == "infeasible":
                    skipped_infeasible += 1
                    continue

                rows.append(
                    self._build_row(domain, task_id, task_config, system_prompt)
                )

        if skipped_missing:
            print(
                f"  ⚠ Skipped {skipped_missing} OSWorld task(s) listed in "
                f"{split_file} with no examples/<domain>/<id>.json",
                flush=True,
            )
        if skipped_infeasible:
            print(
                f"  ⚠ Skipped {skipped_infeasible} infeasible OSWorld task(s) "
                "(include_infeasible=false)",
                flush=True,
            )
        if not rows:
            raise ValueError(
                f"No OSWorld tasks selected under {self.data_dir} ({split_file})."
            )

        self.dataset = Dataset.from_list(rows)
        if shuffle_seed is not None:
            self.dataset = self.dataset.shuffle(seed=shuffle_seed)
        self.val_dataset = None
        print(f"  ✓ Loaded {len(self.dataset)} OSWorld tasks", flush=True)

    def _build_row(
        self,
        domain: str,
        task_id: str,
        task_config: dict[str, Any],
        system_prompt: str,
    ) -> dict[str, Any]:
        instruction = str(task_config.get("instruction", "")).strip()
        related_apps = task_config.get("related_apps", []) or []
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ],
            "task_name":          self.task_name,
            "task_id":            task_id,
            "domain":             domain,
            "instruction":        instruction,
            "related_apps_json":  json.dumps(related_apps),
            "task_config_json":   json.dumps(task_config),
            "max_turns":          self.max_turns,
            "screen_width":       self.screen_width,
            "screen_height":      self.screen_height,
            "pause_after_action": self.pause_after_action,
        }


def osworld_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: Any,
    max_seq_length: int | None,
    idx: int,
) -> DatumSpec:
    """Process an OSWorld row (from ``OSWorldDataset._build_row``).

    The prompt is the system + user message log with no answer turn; it is
    text-only because the desktop is not booted at data-load time. Everything the
    CUA environment/backend needs — the full task_config, related apps, turn
    budget, screen size, and per-action pause — is routed into ``extra_env_info``
    (dict/list fields are carried as JSON strings in the Arrow row and decoded
    here). The CUA rollout injects the per-turn screenshots at rollout time.
    """
    # add_generation_prompt=False: the CUA rollout appends the turn-0 screenshot
    # as the first user observation, and that message carries the generation
    # prompt. Adding it here too would inject an empty assistant turn between the
    # instruction and the first screenshot. OSWorld runs only through the CUA
    # rollout (which always seeds an observation), so this coupling is by design.
    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_eos_token=False,
        add_generation_prompt=False,
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    extra_env_info = {
        "task_id":            datum_dict["task_id"],
        "domain":             datum_dict["domain"],
        "instruction":        datum_dict["instruction"],
        "related_apps":       json.loads(datum_dict["related_apps_json"]),
        "task_config":        json.loads(datum_dict["task_config_json"]),
        "max_turns":          datum_dict["max_turns"],
        "screen_width":       datum_dict["screen_width"],
        "screen_height":      datum_dict["screen_height"],
        "pause_after_action": datum_dict["pause_after_action"],
    }

    loss_multiplier = 1.0
    if max_seq_length is not None and length >= max_seq_length:
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output


# Register at import. OSWorldDataset is imported eagerly by the response-dataset
# registry at startup (before any dataset is loaded / set_processor runs), so the
# processor is available by the time it is referenced. Guard against re-import.
if "osworld_data_processor" not in PROCESSOR_REGISTRY:
    register_processor("osworld_data_processor", osworld_data_processor)
