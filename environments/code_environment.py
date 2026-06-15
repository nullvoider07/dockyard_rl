# SWE-bench-style coding environment, scored via the ubuntu-swe task executor.
#
# The episode is single-shot: the agent emits a unified-diff solution, which
# this environment submits to a task executor (POST /task/submit). The executor
# clones the repo fresh, applies the agent patch, applies the gold test_patch,
# runs the FAIL_TO_PASS / PASS_TO_PASS tests, and returns the verdict. The
# reward is computed from that result by TestRunnerReward; IntegrityReward
# additionally zeroes the reward if the agent patch edits a held-out test file.
#
# Each sample's extra_env_info (produced by swe_bench_data_processor) carries
# repo, base_commit, fail_to_pass, pass_to_pass, test_patch and gold_patch.
# The executor endpoints are configured via env.code.sandbox_urls (or the
# DOCKYARD_SANDBOX_URLS environment variable) and are round-robined across the
# batch; the executor is stateless per task, so no per-episode provisioning is
# needed.

import os
import re
import sys
from typing import Any, NotRequired, Optional, TypedDict, cast

import ray
import torch

from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from dockyard_rl.environments.utils import chunk_list_to_workers
from dockyard_rl.rewards.integrity import IntegrityReward
from dockyard_rl.rewards.interfaces import RewardFunction
from dockyard_rl.rewards.swe_bench_pro import SWEBenchProReward
from dockyard_rl.rewards.test_runner import TestRunnerReward
from dockyard_rl.tool_protocol import CODE_TOOLS, assess_tool_calls

from typing import TYPE_CHECKING

# Deferred: data/interfaces.py
if TYPE_CHECKING:
    from dockyard_rl.data.interfaces import LLMMessageLogType
else:
    try:
        from dockyard_rl.data.interfaces import LLMMessageLogType
    except ImportError:
        LLMMessageLogType = list  # type: ignore


class CodeEnvConfig(TypedDict):
    num_workers: int
    # "binary" → 1.0 when resolved else 0.0; "test_pass_rate" → pass fraction.
    reward_mode: str
    # Which inner reward to build: "swe_bench" (clone + pytest node IDs, default)
    # or "swe_bench_pro" (per-instance Docker image + named-test resolution).
    reward_kind: NotRequired[str]
    # Zero the reward when the agent patch modifies a held-out test file.
    integrity_check: bool
    # Per-task test wall-clock timeout (seconds), enforced executor-side.
    task_timeout: int
    # Task-executor endpoints for /task/submit. Falls back to the
    # DOCKYARD_SANDBOX_URLS environment variable when absent.
    sandbox_urls: NotRequired[list[str]]
    # Bearer token for the executor; falls back to DOCKYARD_SANDBOX_API_TOKEN.
    api_token: NotRequired[Optional[str]]
    # Parse the solution as a structured submit_patch tool call (Hermes
    # <tool_call>) instead of the legacy ```diff fence / <patch> tag. Off →
    # behaviour byte-identical.
    structured_tools: NotRequired[bool]


# Patch extraction
_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL)
_PATCH_TAG_RE  = re.compile(r"<patch>(.*?)</patch>", re.DOTALL)


def _extract_patch(text: str) -> str:
    """Extract a unified-diff solution from an assistant message.

    Priority: ```diff / ```patch fence, then <patch>...</patch>, then the raw
    text when it already looks like a diff. Returns "" when none is found.
    """
    m = _DIFF_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _PATCH_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    stripped = text.strip()
    if stripped.startswith("diff --git") or stripped.startswith("--- "):
        return stripped
    return ""


def _extract_patch_structured(text: str) -> tuple[str, bool]:
    """Extract the patch from a structured ``submit_patch`` tool call.

    Returns ``(patch, invalid)``. ``invalid`` is the #2656 verdict: True when
    the turn is malformed, names an unknown tool, has schema-invalid arguments,
    emits no tool call, or emits more than one call (single action per turn).
    The extracted ``patch`` flows into the scoring path unchanged.
    """
    assessment = assess_tool_calls(text, CODE_TOOLS)
    if assessment.malformed or len(assessment.valid_calls) != 1:
        return "", True
    call = assessment.valid_calls[0]
    if call.name != "submit_patch":
        return "", True
    patch = str(call.arguments["patch"])
    return patch, not patch.strip()


def _resolve_sandbox_urls(cfg: dict) -> list[str]:
    urls = cfg.get("sandbox_urls")
    if not urls:
        raw = os.environ.get("DOCKYARD_SANDBOX_URLS", "")
        urls = [u.strip() for u in raw.split(",") if u.strip()]
    if not urls:
        raise ValueError(
            "No task-executor endpoints configured. Set env.code.sandbox_urls "
            "or the DOCKYARD_SANDBOX_URLS environment variable."
        )
    return list(urls)


@ray.remote  # pragma: no cover
class SWEScoringWorker:
    """Worker that scores extracted patches against the task executor.

    Holds the composite reward (IntegrityReward wrapping TestRunnerReward) and
    evaluates a chunk of trajectories. Workers are stateless across episodes;
    the executor clones a fresh repo for every submitted task.
    """

    def __init__(
        self,
        reward_mode: str,
        integrity_check: bool,
        api_token: Optional[str] = None,
        reward_kind: str = "swe_bench",
    ) -> None:
        inner: RewardFunction
        if reward_kind == "swe_bench_pro":
            inner = SWEBenchProReward(reward_mode=reward_mode, api_token=api_token)
        else:
            inner = TestRunnerReward(reward_mode=reward_mode, api_token=api_token)
        self._reward = IntegrityReward(inner) if integrity_check else inner

    def score(self, trajectories: list[dict]) -> list[dict]:
        results: list[dict] = []
        for trajectory in trajectories:
            try:
                outcome = self._reward(trajectory)
                results.append(
                    {
                        "reward": float(outcome.reward),
                        "status": outcome.status,
                        "failure_reason": outcome.failure_reason,
                    }
                )
            except Exception as exc:  # never let one task crash the batch
                results.append(
                    {"reward": 0.0, "status": "execution_error", "failure_reason": str(exc)}
                )
        return results


@ray.remote  # pragma: no cover
class CodeEnvironment(EnvironmentInterface):
    """Single-shot SWE-bench environment backed by the ubuntu-swe task executor.

    The agent's response is parsed for a unified-diff solution, which is scored
    by submitting it (with the gold test_patch and the FAIL_TO_PASS/PASS_TO_PASS
    node IDs) to a task executor. The episode ends after this single scoring
    step.
    """

    def __init__(self, cfg: CodeEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.reward_mode = cfg.get("reward_mode", "binary")
        self.reward_kind = cfg.get("reward_kind", "swe_bench")
        self.integrity_check = cfg.get("integrity_check", True)
        self.task_timeout = int(cfg.get("task_timeout", 300))
        self.structured_tools = bool(cfg.get("structured_tools", False))
        self.sandbox_urls = _resolve_sandbox_urls(cast(dict, cfg))
        self.api_token = cfg.get("api_token")

        self.workers = cast(list[Any], [
            SWEScoringWorker.options(
                runtime_env={"py_executable": sys.executable}
            ).remote(
                reward_mode=self.reward_mode,
                integrity_check=self.integrity_check,
                api_token=self.api_token,
                reward_kind=self.reward_kind,
            )
            for _ in range(self.num_workers)
        ])

    def _build_trajectory(
        self, patch: str, meta: dict, sandbox_url: str
    ) -> dict:
        repo = meta.get("repo", "")
        return {
            "sandbox_url":  sandbox_url,
            "task_id":      meta.get("instance_id", ""),
            "repo_url":     f"https://github.com/{repo}.git" if repo else "",
            "base_commit":  meta.get("base_commit", "HEAD"),
            "patch":        patch,
            "test_patch":   meta.get("test_patch", ""),
            "fail_to_pass": meta.get("fail_to_pass", []),
            "pass_to_pass": meta.get("pass_to_pass", []),
            "gold_patch":   meta.get("gold_patch", ""),
            "timeout":      self.task_timeout,
            # SWE-bench Pro (image mode) fields; empty/ignored for the clone path.
            "image":               meta.get("image", ""),
            "repo_language":       meta.get("repo_language", ""),
            "before_repo_set_cmd": meta.get("before_repo_set_cmd", ""),
            "selected_test_files": meta.get("selected_test_files", []),
            "run_script":          meta.get("run_script", ""),
            "parser_script":       meta.get("parser_script", ""),
        }

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[dict],
    ) -> EnvironmentReturn:
        """Extract each agent patch, score it, and end the episode."""
        responses = [
            str(ml[-1]["content"]) if ml else "" for ml in message_log_batch
        ]
        # Environment-authoritative invalid-action verdict (#2656): no patch
        # could be extracted from the response. The rollout loop reads and pops
        # this key per turn.
        if self.structured_tools:
            extracted = [_extract_patch_structured(resp) for resp in responses]
            patches = [p for p, _ in extracted]
            for (_, invalid), meta in zip(extracted, metadata):
                meta["invalid_action"] = invalid
        else:
            patches = [_extract_patch(resp) for resp in responses]
            for patch, meta in zip(patches, metadata):
                meta["invalid_action"] = not patch.strip()
        trajectories = [
            self._build_trajectory(
                patch,
                meta,
                self.sandbox_urls[i % len(self.sandbox_urls)],
            )
            for i, (patch, meta) in enumerate(zip(patches, metadata))
        ]

        chunked = chunk_list_to_workers(trajectories, self.num_workers)
        futures = [
            cast(Any, self.workers[i].score).remote(chunk)
            for i, chunk in enumerate(chunked)
            if chunk
        ]
        results: list[dict] = []
        for chunk_result in ray.get(futures):
            results += chunk_result

        rewards = torch.tensor([r["reward"] for r in results], dtype=torch.float32)
        observations = [
            {"role": "user", "content": r["failure_reason"] or "resolved"}
            for r in results
        ]
        terminateds = torch.ones(len(results), dtype=torch.bool)
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=cast(Any, next_stop_strings),
            rewards=rewards,
            terminateds=terminateds,
            answers=None,
        )

    def shutdown(self) -> None:
        for worker in self.workers:
            ray.kill(worker)

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        """Compute batch-level metrics after all rollouts complete."""
        rewards = (
            batch["rewards"]
            if batch["rewards"].ndim == 1
            else batch["rewards"][:, 0]
        )
        if "is_end" in batch:
            rewards = rewards * batch["is_end"]

        metrics = {
            "accuracy": rewards.mean().item(),
            "resolved_rate": (rewards >= 1.0).float().mean().item(),
            "num_problems_in_batch": int(rewards.shape[0]),
        }
        return batch, metrics
