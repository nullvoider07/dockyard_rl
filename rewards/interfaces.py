from __future__ import annotations

import abc
from typing import NamedTuple

class RewardVerificationResult(NamedTuple):
    """Carries the outcome of a single reward computation."""

    reward: float
    status: str  # "ok" | "integrity_failure" | "execution_error"
    evidence_hash: str | None
    failure_reason: str | None

class RewardFunction(abc.ABC):
    """Abstract base class for all reward functions in dockyard-rl.

    Every concrete reward function receives a trajectory dict and returns a
    RewardVerificationResult.  The scalar reward is always accessible via
    result.reward; callers that only need the number can do float(result.reward).

    Trajectory dict contract (SWE-bench scoring):
        sandbox_url  (str)  — base URL of a task executor (/task/submit)
        task_id      (str)  — unique identifier for this episode/task
        repo_url     (str)  — git URL the executor clones
        base_commit  (str)  — commit checked out before patches apply
        patch        (str)  — the agent's extracted solution diff
        test_patch   (str)  — gold test diff, applied by the executor
        fail_to_pass (list) — test node IDs that must pass
        pass_to_pass (list) — test node IDs that must stay passing
        gold_patch   (str)  — reference solution diff (patch-similarity only)
    """

    @abc.abstractmethod
    def __call__(self, trajectory: dict) -> RewardVerificationResult:
        """Compute reward for a completed trajectory.

        Args:
            trajectory: Dict carrying the fields above (at minimum sandbox_url
                        and the repo/patch/test metadata the reward needs).

        Returns:
            RewardVerificationResult with reward, status, evidence_hash,
            and failure_reason populated.
        """