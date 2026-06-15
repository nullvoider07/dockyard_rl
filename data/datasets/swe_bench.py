"""SWE-Bench dataset loader for Project Dockyard RL training.

Trainer-side only.  Reads SWE-bench task specifications and presents them
as rows consumed by the GRPO dataloader, following the same convention as
every other dataset in ``data/datasets/response_datasets/``:

  * ``SWEBenchDataset`` subclasses :class:`RawDataset`, loads the raw HF
    dataset into ``self.dataset`` and maps each row into a ``messages`` +
    metadata record via :meth:`format_data`.
  * Tokenisation is *not* done here.  ``load_response_dataset`` calls
    ``set_task_spec`` / ``set_processor`` (inherited from ``RawDataset``)
    and the configured processor turns each row into a ``DatumSpec``.
  * For SWE-bench the matching processor is ``swe_bench_data_processor``
    (see ``data/processors.py``); set ``processor: swe_bench_data_processor``
    in the dataset config so the test metadata reaches the reward function.

This file has nothing to do with the sandbox environment.  Environment
setup (cloning repos, installing deps, configuring test runners) is
handled by scripts/envs/swe_bench_setup.sh, which runs inside ubuntu-swe
containers at episode start time via the task executor REST API.

Dataset schema (princeton-nlp/SWE-bench or compatible):
    instance_id          str   — unique task identifier
    repo                 str   — GitHub repo (e.g. "django/django")
    base_commit          str   — commit SHA the agent starts from
    problem_statement    str   — natural-language bug description
    hints_text           str   — optional human hints (may be empty)
    test_patch           str   — the ground-truth test diff (held out)
    patch                str   — the ground-truth fix diff (held out)
    FAIL_TO_PASS         str   — JSON list of tests that must go fail→pass
    PASS_TO_PASS         str   — JSON list of tests that must stay passing
    environment_setup_commit str — commit to use for env setup (may differ)

The agent never sees test_patch or patch — those are used only by the
reward function (rewards/test_runner.py) to verify the agent's solution,
which reads fail_to_pass / pass_to_pass from the DatumSpec's
``extra_env_info``.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from datasets import load_dataset
from dockyard_rl.data.datasets.raw_dataset import RawDataset

# Prompt template
# The system prompt tells the agent what tools and workflow to use.
# Kept in one place so it's easy to swap without touching dataset logic.
_SYSTEM_PROMPT = """\
You are an expert software engineer. You are given a bug report for a software
repository. Produce a single unified diff (a git patch) that fixes the bug so
the project's tests pass.

Output requirements:
- Reply with exactly one unified diff inside a ```diff code fence.
- The diff must apply cleanly from the repository root with `git apply`.
- Modify only the source files needed for the fix; do not modify test files.
- Put no prose outside the code fence.

Reason through the fix, then output only the diff.\
"""

_USER_TEMPLATE = """\
<problem_statement>
{problem_statement}
</problem_statement>

<repository>
{repo} @ {base_commit}
</repository>
{hints_block}
Your task: fix the bug described above so the failing tests pass without \
breaking any currently passing tests.\
"""

_HINTS_BLOCK = """\

<hints>
{hints_text}
</hints>
"""

def _parse_test_list(value: Any) -> list[str]:
    """Normalise a FAIL_TO_PASS / PASS_TO_PASS field into a list of test IDs.

    Different SWE-bench mirrors expose these either as a JSON-encoded string
    (the canonical ``princeton-nlp`` releases) or as a native list.  Handle
    both, and degrade gracefully on anything unexpected.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            # Not JSON — treat the raw string as a single test id.
            return [value]
        return parsed if isinstance(parsed, list) else [str(parsed)]
    return []

class SWEBenchDataset(RawDataset):
    """SWE-bench task instances exposed as ``messages`` + metadata rows.

    Each mapped row contains:
        messages                 list  — system + user prompt (no answer turn)
        task_name                str   — always "swe_bench"
        instance_id              str   — unique task identifier
        repo                     str   — GitHub repo slug
        base_commit              str   — starting commit SHA
        environment_setup_commit str   — commit to use for env setup
        fail_to_pass             list  — test IDs that must go fail→pass
        pass_to_pass             list  — test IDs that must stay passing
        test_patch               str   — held-out gold test diff (introduces
                                         the FAIL_TO_PASS tests at scoring time)
        gold_patch               str   — held-out reference fix diff (used only
                                         for patch-similarity scoring)

    The ``fail_to_pass`` / ``pass_to_pass`` / commit fields are carried into
    the ``DatumSpec``'s ``extra_env_info`` by ``swe_bench_data_processor`` so
    the reward function can verify the agent's solution without ever exposing
    the held-out test or fix patches to the agent.

    Args:
        hf_dataset_name: HuggingFace dataset name or local path. Point this at
            ``princeton-nlp/SWE-bench_Verified`` for the Verified split.
        split:           Dataset split to load.
        instance_ids:    Optional list of instance_ids to restrict to.
                         None = load all instances.
        shuffle_seed:    If set, shuffle the dataset with this seed.
    """

    def __init__(
        self,
        hf_dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        split: str = "test",
        instance_ids: Optional[list[str]] = None,
        shuffle_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.task_name = "swe_bench"

        print(
            f"Loading SWE-bench dataset: {hf_dataset_name} (split={split})",
            flush=True,
        )
        raw = load_dataset(hf_dataset_name, split=split)

        if instance_ids:
            id_set = set(instance_ids)
            raw = raw.filter(lambda x: x["instance_id"] in id_set)
            print(
                f"  Subset filter applied: {len(raw)} instances remain",
                flush=True,
            )

        if shuffle_seed is not None:
            raw = raw.shuffle(seed=shuffle_seed)

        self.dataset = raw.map(
            self.format_data,
            remove_columns=raw.column_names,
        )

        # `self.val_dataset` is used (not None) only when current dataset is
        # used for both training and validation.
        self.val_dataset = None
        print(
            f"  ✓ Loaded {len(self.dataset)} SWE-bench instances",
            flush=True,
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        repo = data["repo"]
        base_commit = data["base_commit"]
        problem_statement = (data.get("problem_statement") or "").strip()
        hints_text = (data.get("hints_text") or "").strip()

        hints_block = (
            _HINTS_BLOCK.format(hints_text=hints_text) if hints_text else ""
        )
        user_content = _USER_TEMPLATE.format(
            problem_statement = problem_statement,
            repo              = repo,
            base_commit       = base_commit[:12],  # short SHA
            hints_block       = hints_block,
        )

        return {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            "task_name":                self.task_name,
            "instance_id":              data["instance_id"],
            "repo":                     repo,
            "base_commit":              base_commit,
            "environment_setup_commit": data.get("environment_setup_commit")
            or base_commit,
            "fail_to_pass":             _parse_test_list(data.get("FAIL_TO_PASS")),
            "pass_to_pass":             _parse_test_list(data.get("PASS_TO_PASS")),
            # Held-out: never shown to the agent (carried in extra_env_info).
            # test_patch introduces the FAIL_TO_PASS tests; gold_patch is the
            # reference solution used only for patch-similarity scoring.
            "test_patch":               data.get("test_patch") or "",
            "gold_patch":               data.get("patch") or "",
        }
