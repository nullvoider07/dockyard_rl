from __future__ import annotations

import hashlib
import shlex
from typing import Optional

from dockyard_rl.rewards.interfaces import RewardFunction, RewardVerificationResult
from dockyard_rl.sandbox import TaskExecutorError, TaskSpec, run_task

_SUPPORTED_FRAMEWORKS = ("pytest",)


def _build_pytest_command(node_ids: list[str]) -> str:
    targets = " ".join(shlex.quote(n) for n in node_ids)
    return f"python3 -m pytest -p no:cacheprovider -q {targets}".strip()


def _evidence_hash(text: str) -> Optional[str]:
    return hashlib.sha256(text.encode()).hexdigest() if text else None


class TestRunnerReward(RewardFunction):
    """Score a SWE-bench solution via the task executor's /task/submit API.

    The agent's ``patch`` is applied to a fresh clone, the executor applies the
    gold ``test_patch``, and the ``FAIL_TO_PASS ∪ PASS_TO_PASS`` node IDs are
    run. A task is resolved iff the test command exits 0 with no failures and at
    least the expected number of tests pass.

    Args:
        reward_mode: "binary" → 1.0 when resolved else 0.0;
                     "test_pass_rate" → passed / expected, clamped to [0, 1].
        framework:   Test framework whose command is built. Only "pytest" is
                     supported (SWE-bench); other values raise.
        api_token:   Bearer token for the executor; defaults to the
                     DOCKYARD_SANDBOX_API_TOKEN environment variable.
    """

    def __init__(
        self,
        reward_mode: str = "binary",
        framework: str = "pytest",
        api_token: Optional[str] = None,
    ) -> None:
        if reward_mode not in ("binary", "test_pass_rate"):
            raise ValueError(
                f"Unsupported reward_mode '{reward_mode}'. "
                "Use 'binary' or 'test_pass_rate'."
            )
        if framework not in _SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"Unsupported framework '{framework}'. "
                f"Supported: {list(_SUPPORTED_FRAMEWORKS)}"
            )
        self._reward_mode = reward_mode
        self._framework = framework
        self._api_token = api_token

    def __call__(self, trajectory: dict) -> RewardVerificationResult:
        node_ids = list(trajectory.get("fail_to_pass", [])) + list(
            trajectory.get("pass_to_pass", [])
        )
        if not node_ids:
            return RewardVerificationResult(
                reward=0.0,
                status="execution_error",
                evidence_hash=None,
                failure_reason="trajectory has no fail_to_pass/pass_to_pass node IDs",
            )

        spec: TaskSpec = {
            "repo_url":        trajectory["repo_url"],
            "base_commit":     trajectory.get("base_commit", "HEAD"),
            "patch":           trajectory.get("patch", ""),
            "test_patch":      trajectory.get("test_patch", ""),
            "test_command":    _build_pytest_command(node_ids),
            "reference_patch": trajectory.get("gold_patch", ""),
            "capture_diff":    True,
            "timeout":         int(trajectory.get("timeout", 300)),
        }

        try:
            result = run_task(
                trajectory["sandbox_url"], spec, api_token=self._api_token
            )
        except TaskExecutorError as exc:
            return RewardVerificationResult(
                reward=0.0,
                status="execution_error",
                evidence_hash=None,
                failure_reason=str(exc),
            )

        if result["status"] != "completed":
            return RewardVerificationResult(
                reward=0.0,
                status="execution_error",
                evidence_hash=None,
                failure_reason=(result.get("stderr") or "task did not complete")[:512],
            )
        if not result.get("test_patch_applied", True):
            return RewardVerificationResult(
                reward=0.0,
                status="execution_error",
                evidence_hash=None,
                failure_reason="gold test_patch failed to apply",
            )

        evidence = _evidence_hash(result.get("patch_diff") or "")

        # An agent patch that did not apply is a scored zero, not an error.
        if not result.get("patch_applied", True):
            return RewardVerificationResult(
                reward=0.0,
                status="ok",
                evidence_hash=evidence,
                failure_reason="agent patch did not apply",
            )

        passed = result.get("tests_passed") or 0
        failed = result.get("tests_failed") or 0
        exit_code = result.get("exit_code", -1)
        expected = len(node_ids)

        resolved = exit_code == 0 and failed == 0 and passed >= expected
        if self._reward_mode == "binary":
            reward = 1.0 if resolved else 0.0
        else:
            reward = max(0.0, min(1.0, passed / expected))

        failure_reason = (
            None
            if resolved
            else f"unresolved: passed={passed}/{expected} failed={failed} exit={exit_code}"
        )
        return RewardVerificationResult(
            reward=reward,
            status="ok",
            evidence_hash=evidence,
            failure_reason=failure_reason,
        )
