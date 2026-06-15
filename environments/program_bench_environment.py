# ProgramBench environment — multi-turn cleanroom reconstruction. Built on
# MultiTurnSessionEnvironment: the agent probes an execute-only reference binary
# (./executable) in a network-isolated session (cap-drop SYS_PTRACE) and writes
# source + compile.sh. On completion the agent's /workspace is exported as the
# submission and graded by ProgramBenchReward, which rebuilds it from scratch in
# a clean image and runs the held-out behavioural pytest branches.

from typing import NotRequired, Optional, TypedDict

import ray

from dockyard_rl.environments.multi_turn_session_env import (
    MultiTurnSessionEnvironment,
)
from dockyard_rl.rewards.program_bench import ProgramBenchReward
from dockyard_rl.sandbox import SessionStartSpec, delete_session, session_export


class ProgramBenchEnvConfig(TypedDict):
    # "test_pass_rate" → resolved/total behavioural pass-rate; "binary" → all-pass.
    reward_mode: str
    # Default per-episode turn budget when a task row omits max_turns.
    max_turns: int
    # Max bytes of stdout/stderr returned per command observation.
    output_limit: NotRequired[int]
    # Run the grading container offline (cleanroom default True).
    grading_block_network: NotRequired[bool]
    # Actor concurrency for servicing parallel step() calls (async rollout path).
    max_concurrency: NotRequired[int]
    # Task-executor endpoints; falls back to DOCKYARD_SANDBOX_URLS.
    sandbox_urls: NotRequired[list[str]]
    # Bearer token; falls back to DOCKYARD_SANDBOX_API_TOKEN.
    api_token: NotRequired[Optional[str]]


@ray.remote  # pragma: no cover
class ProgramBenchEnvironment(MultiTurnSessionEnvironment):
    """Multi-turn ProgramBench environment (cleanroom rebuild + behavioural tests)."""

    def _setup(self, cfg: dict) -> None:
        grading_block_network = bool(cfg.get("grading_block_network", True))
        self._reward = ProgramBenchReward(
            reward_mode=cfg.get("reward_mode", "test_pass_rate"),
            api_token=self.api_token,
            block_network=grading_block_network,
        )

    def _start_spec(self, meta: dict) -> SessionStartSpec:
        # Inference-phase cleanroom policy: no network (can't fetch source) and
        # drop SYS_PTRACE (can't reverse-engineer the execute-only binary).
        return {
            "mode":          "image",
            "image":         meta.get("image", ""),
            "block_network": True,
            "cap_drop":      ["SYS_PTRACE"],
            "pull_timeout":  int(meta.get("grading_timeout_sec", 1800)),
        }

    def _finish_and_score(self, url: str, session_id: str, meta: dict) -> tuple[float, str]:
        """Export the agent submission, tear the session down, grade host-side."""
        try:
            submission = session_export(url, session_id, api_token=self.api_token)
        finally:
            # Done with the agent container; grading runs in a fresh image task.
            delete_session(url, session_id, api_token=self.api_token)
        traj = {
            "sandbox_url":   url,
            "image":         meta.get("image", ""),
            "submission_tar": submission,
            "branches":      meta.get("branches") or {},
            "timeout":       int(meta.get("grading_timeout_sec", 3600)),
        }
        outcome = self._reward(traj)
        verdict = (
            "resolved"
            if outcome.reward >= 1.0 and outcome.status == "ok"
            else (outcome.failure_reason or "unresolved")
        )
        return float(outcome.reward), f"[grading] {verdict}"
