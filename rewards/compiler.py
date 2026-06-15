from __future__ import annotations

from typing import Any, Optional

from dockyard_rl.rewards.interfaces import RewardFunction, RewardVerificationResult
from dockyard_rl.sandbox import TaskExecutorError, TaskSpec, run_task

# Per-language build command run as the task's test_command. Python has no
# separate build step (import-time errors surface in the test run instead).
_LANGUAGE_COMMANDS: dict[str, dict[str, Any]] = {
    "python": {"build_cmd": None,                       "timeout": 10},
    "rust":   {"build_cmd": "cargo build 2>&1",          "timeout": 120},
    "go":     {"build_cmd": "go build ./... 2>&1",       "timeout": 60},
    "java":   {"build_cmd": "find . -name '*.java' -print0 | xargs -0 javac 2>&1", "timeout": 60},
    "c":      {"build_cmd": "find . -name '*.c' -print0 | xargs -0 gcc -o /tmp/_dockyard_c_out 2>&1", "timeout": 60},
    "cpp":    {"build_cmd": "find . -name '*.cpp' -print0 | xargs -0 g++ -o /tmp/_dockyard_cpp_out 2>&1", "timeout": 60},
    "dotnet": {"build_cmd": "dotnet build 2>&1",         "timeout": 120},
}


class CompilerReward(RewardFunction):
    """Reward a build's success via the task executor.

    Submits the agent's patch with the per-language build command as the
    task's test_command. Reward is +1.0 on a clean build (exit 0), -1.0 on a
    build failure. Python has no build step and always scores +1.0.

    Args:
        language:  One of "python", "rust", "go", "java", "c", "cpp", "dotnet".
        api_token: Bearer token for the executor; defaults to the
                   DOCKYARD_SANDBOX_API_TOKEN environment variable.
    """

    def __init__(self, language: str, api_token: Optional[str] = None) -> None:
        language = language.lower()
        if language not in _LANGUAGE_COMMANDS:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Supported: {sorted(_LANGUAGE_COMMANDS)}"
            )
        self._language = language
        self._cfg = _LANGUAGE_COMMANDS[language]
        self._api_token = api_token

    def __call__(self, trajectory: dict) -> RewardVerificationResult:
        build_cmd = self._cfg["build_cmd"]
        if build_cmd is None:
            return RewardVerificationResult(
                reward=1.0, status="ok", evidence_hash=None, failure_reason=None
            )

        spec: TaskSpec = {
            "repo_url":     trajectory["repo_url"],
            "base_commit":  trajectory.get("base_commit", "HEAD"),
            "patch":        trajectory.get("patch", ""),
            "test_command": build_cmd,
            "timeout":      int(self._cfg["timeout"]),
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
        if not result.get("patch_applied", True):
            return RewardVerificationResult(
                reward=-1.0,
                status="ok",
                evidence_hash=None,
                failure_reason="agent patch did not apply",
            )

        exit_code = result.get("exit_code", -1)
        if exit_code == 0:
            return RewardVerificationResult(
                reward=1.0, status="ok", evidence_hash=None, failure_reason=None
            )
        return RewardVerificationResult(
            reward=-1.0,
            status="ok",
            evidence_hash=None,
            failure_reason=f"build failed (exit {exit_code})",
        )
