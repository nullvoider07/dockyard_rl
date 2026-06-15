"""SWE-bench Pro dataset loader for Project Dockyard RL training.

Trainer-side only. Loads ``ScaleAI/SWE-bench_Pro`` and presents each instance
as a ``messages`` + metadata row, following the same convention as
``swe_bench.py``. Unlike SWE-bench Lite, Pro is scored inside a prebuilt
per-instance Docker image (``{registry}:{dockerhub_tag}``) running that
instance's own ``run_script.sh`` + ``parser.py`` — so this loader also embeds
that per-instance harness (fetched by ``scripts/cluster/swe_bench_pro_setup.sh``)
into each row's metadata. The reward (``rewards/swe_bench_pro.py``) consumes it
from ``extra_env_info``; the agent never sees it.

Dataset schema (ScaleAI/SWE-bench_Pro, ``test`` split, 731 rows):
    instance_id                str  — unique id; matches the run_scripts/<id> dir
    repo                       str  — "owner/name"
    base_commit                str  — commit the agent starts from
    patch                      str  — gold fix diff (held out)
    test_patch                 str  — gold test diff (held out; integrity check)
    problem_statement          str  — task description (shown to agent)
    requirements               str  — required behaviour (shown to agent)
    interface                  str  — relevant code interface (shown to agent)
    repo_language              str  — js / python / go / ...
    fail_to_pass               str|list — named tests that must go fail→pass
    pass_to_pass               str|list — named tests that must stay passing
    before_repo_set_cmd        str  — multi-line; its LAST line checks out the
                                       gold test files from the solution commit
    selected_test_files_to_run str  — JSON array of test files to run
    dockerhub_tag              str  — per-instance image tag

The agent prompt is built only from problem_statement / requirements /
interface / repo@commit — never from the held-out tests, patches, or harness.
"""

from __future__ import annotations

import ast
import json
import os
from typing import Any, Optional

from datasets import load_dataset
from dockyard_rl.data.datasets.raw_dataset import RawDataset

_SYSTEM_PROMPT = """\
You are an expert software engineer. You are given a software engineering task
for a repository: a problem description, the requirements your change must
satisfy, and the relevant code interface. Produce a single unified diff (a git
patch) that implements the change so the project's tests pass.

Output requirements:
- Reply with exactly one unified diff inside a ```diff code fence.
- The diff must apply cleanly from the repository root with `git apply`.
- Modify only the source files needed; do not modify test files.
- Put no prose outside the code fence.

Reason through the change, then output only the diff.\
"""

_USER_TEMPLATE = """\
<problem_statement>
{problem_statement}
</problem_statement>
{requirements_block}{interface_block}
<repository>
{repo} @ {base_commit} (language: {repo_language})
</repository>

Your task: implement the change described above so the project's tests pass \
without breaking any currently passing tests.\
"""

_REQUIREMENTS_BLOCK = """
<requirements>
{requirements}
</requirements>
"""

_INTERFACE_BLOCK = """
<interface>
{interface}
</interface>
"""

def _to_list(value: Any) -> list[str]:
    """Normalise a fail/pass/selected-files field into a list of strings.

    Pro stores these either as native lists or as Python-repr strings of lists
    (single quotes, embedded apostrophes), so ``ast.literal_eval`` is tried
    before ``json.loads``; a bare unparseable string becomes one element.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(s)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, (list, tuple)):
                return [str(v) for v in parsed]
        return [s]
    return []


def _harness_paths(scripts_dir: str, instance_id: str) -> tuple[str, str]:
    base = os.path.join(scripts_dir, instance_id)
    return os.path.join(base, "run_script.sh"), os.path.join(base, "parser.py")


def _has_harness(scripts_dir: str, instance_id: str) -> bool:
    run_script, parser_script = _harness_paths(scripts_dir, instance_id)
    return os.path.isfile(run_script) and os.path.isfile(parser_script)


class SWEBenchProDataset(RawDataset):
    """SWE-bench Pro instances exposed as ``messages`` + metadata rows.

    Args:
        hf_dataset_name: HuggingFace dataset name or local path.
        split:           Dataset split to load.
        scripts_dir:     Directory of per-instance harness dirs produced by
                         ``swe_bench_pro_setup.sh``. Defaults to the
                         ``DOCKYARD_SWE_BENCH_PRO_SCRIPTS_DIR`` env var. Instances
                         without a harness dir are dropped (with a count).
        image_registry:  Registry/repo prefixed to ``dockerhub_tag`` to form the
                         full image ref. Default ``jefzda/sweap-images``.
        instance_ids:    Optional subset of instance_ids.
        shuffle_seed:    If set, shuffle with this seed.
    """

    def __init__(
        self,
        hf_dataset_name: str = "ScaleAI/SWE-bench_Pro",
        split: str = "test",
        scripts_dir: Optional[str] = None,
        image_registry: str = "jefzda/sweap-images",
        instance_ids: Optional[list[str]] = None,
        shuffle_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.task_name = "swe_bench_pro"
        self.scripts_dir = scripts_dir or os.environ.get(
            "DOCKYARD_SWE_BENCH_PRO_SCRIPTS_DIR", ""
        )
        self.image_registry = image_registry.rstrip(":/")

        if not self.scripts_dir or not os.path.isdir(self.scripts_dir):
            raise ValueError(
                "SWE-bench Pro requires the per-instance harness. Set "
                "scripts_dir (or DOCKYARD_SWE_BENCH_PRO_SCRIPTS_DIR) to the "
                "directory populated by scripts/cluster/swe_bench_pro_setup.sh; "
                f"got {self.scripts_dir!r}."
            )

        print(
            f"Loading SWE-bench Pro dataset: {hf_dataset_name} (split={split})",
            flush=True,
        )
        raw = load_dataset(hf_dataset_name, split=split)

        if instance_ids:
            id_set = set(instance_ids)
            raw = raw.filter(lambda x: x["instance_id"] in id_set)

        # Drop instances whose harness was not provisioned — they cannot be
        # scored in image mode. Report how many were dropped.
        n_before = len(raw)
        raw = raw.filter(lambda x: _has_harness(self.scripts_dir, x["instance_id"]))
        dropped = n_before - len(raw)
        if dropped:
            print(
                f"  ⚠ Dropped {dropped} instance(s) with no harness dir under "
                f"{self.scripts_dir}",
                flush=True,
            )

        if shuffle_seed is not None:
            raw = raw.shuffle(seed=shuffle_seed)

        self.dataset = raw.map(self.format_data, remove_columns=raw.column_names)
        self.val_dataset = None
        print(f"  ✓ Loaded {len(self.dataset)} SWE-bench Pro instances", flush=True)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        repo = data["repo"]
        base_commit = data["base_commit"]
        instance_id = data["instance_id"]
        repo_language = (data.get("repo_language") or "").strip() or "unknown"

        problem_statement = (data.get("problem_statement") or "").strip()
        requirements = (data.get("requirements") or "").strip()
        interface = (data.get("interface") or "").strip()

        requirements_block = (
            _REQUIREMENTS_BLOCK.format(requirements=requirements) if requirements else ""
        )
        interface_block = (
            _INTERFACE_BLOCK.format(interface=interface) if interface else ""
        )
        user_content = _USER_TEMPLATE.format(
            problem_statement  = problem_statement,
            requirements_block = requirements_block,
            interface_block    = interface_block,
            repo               = repo,
            base_commit        = base_commit[:12],
            repo_language      = repo_language,
        )

        run_script_path, parser_path = _harness_paths(self.scripts_dir, instance_id)
        with open(run_script_path, encoding="utf-8") as fh:
            run_script = fh.read()
        with open(parser_path, encoding="utf-8") as fh:
            parser_script = fh.read()

        return {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            "task_name":           self.task_name,
            "instance_id":         instance_id,
            "repo":                repo,
            "base_commit":         base_commit,
            "repo_language":       repo_language,
            # Full per-instance image reference scored in image mode.
            "image":               f"{self.image_registry}:{data['dockerhub_tag']}",
            # Held-out scoring inputs (never shown to the agent), routed into
            # extra_env_info by swe_bench_pro_data_processor.
            "before_repo_set_cmd": data.get("before_repo_set_cmd") or "",
            "selected_test_files": _to_list(data.get("selected_test_files_to_run")),
            "fail_to_pass":        _to_list(data.get("fail_to_pass")),
            "pass_to_pass":        _to_list(data.get("pass_to_pass")),
            "test_patch":          data.get("test_patch") or "",
            "gold_patch":          data.get("patch") or "",
            "run_script":          run_script,
            "parser_script":       parser_script,
        }
