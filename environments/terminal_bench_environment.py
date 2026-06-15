# Terminal-Bench 2.1 environment — multi-turn terminal episodes graded on the
# final container state. Built on MultiTurnSessionEnvironment: the agent drives a
# long-lived session container; on completion or budget exhaustion the held-out
# tests/ are late-injected (executor /session/finish) and run, and the CTRF
# report is parsed by TerminalBenchReward.

from typing import NotRequired, Optional, TypedDict

import ray

from dockyard_rl.environments.multi_turn_session_env import (
    MultiTurnSessionEnvironment,
)
from dockyard_rl.rewards.terminal_bench import TerminalBenchReward
from dockyard_rl.sandbox import SessionStartSpec, session_finish


class TerminalBenchEnvConfig(TypedDict):
    # "binary" → 1.0 when the verifier passes else 0.0; "test_pass_rate" → frac.
    reward_mode: str
    # Default per-episode turn budget when a task row omits max_turns.
    max_turns: int
    # Honour task.toml allow_internet=false by running the container with no
    # network. Default False: the verifier installs uv from the network, so a
    # network-off container breaks grading (see module/handoff notes).
    respect_allow_internet: NotRequired[bool]
    # Max bytes of stdout/stderr returned per command observation.
    output_limit: NotRequired[int]
    # Actor concurrency for servicing parallel step() calls (async rollout path).
    max_concurrency: NotRequired[int]
    # Task-executor endpoints; falls back to DOCKYARD_SANDBOX_URLS.
    sandbox_urls: NotRequired[list[str]]
    # Bearer token; falls back to DOCKYARD_SANDBOX_API_TOKEN.
    api_token: NotRequired[Optional[str]]


@ray.remote  # pragma: no cover
class TerminalBenchEnvironment(MultiTurnSessionEnvironment):
    """Multi-turn Terminal-Bench environment (final-container-state grading)."""

    def _setup(self, cfg: dict) -> None:
        self.respect_allow_internet = bool(cfg.get("respect_allow_internet", False))
        self._reward = TerminalBenchReward(reward_mode=self.reward_mode)

    def _start_spec(self, meta: dict) -> SessionStartSpec:
        block_network = (
            (not bool(meta.get("allow_internet", True)))
            if self.respect_allow_internet
            else False
        )
        container_env = meta.get("container_env") or {}
        if meta.get("mode") == "build":
            return {
                "mode":          "build",
                "dockerfile":    meta.get("dockerfile", ""),
                "build_context": meta.get("build_context") or {},
                "env":           container_env,
                "block_network": block_network,
                "build_timeout": int(meta.get("build_timeout_sec", 1800)),
            }
        return {
            "mode":          "image",
            "image":         meta.get("image", ""),
            "env":           container_env,
            "block_network": block_network,
            "pull_timeout":  int(meta.get("build_timeout_sec", 1800)),
        }

    def _finish_and_score(self, url: str, session_id: str, meta: dict) -> tuple[float, str]:
        """Run the verifier, parse the CTRF report, return (reward, verdict)."""
        result_files = meta.get("result_files") or []
        fr = session_finish(
            url,
            session_id,
            harness_files=meta.get("tests") or {},
            harness_mount=meta.get("harness_mount") or "/tests",
            test_command=meta.get("test_command"),
            result_files=result_files,
            timeout=int(meta.get("verifier_timeout_sec", 900)),
            api_token=self.api_token,
        )
        files = fr.get("result_files") or {}
        ctrf_path = next((p for p in result_files if p.endswith("ctrf.json")), None)
        reward_path = next((p for p in result_files if p.endswith("reward.txt")), None)
        verifier_error = fr.get("error")
        if not verifier_error and fr.get("timed_out"):
            verifier_error = "verifier timed out"
        traj = {
            "ctrf":               files.get(ctrf_path) if ctrf_path else None,
            "reward_txt":         files.get(reward_path) if reward_path else None,
            "verifier_exit_code": fr.get("exit_code"),
            "verifier_error":     verifier_error,
        }
        outcome = self._reward(traj)
        verdict = (
            "resolved"
            if outcome.reward >= 1.0 and outcome.status == "ok"
            else (outcome.failure_reason or "unresolved")
        )
        return float(outcome.reward), f"[verifier] {verdict}"
