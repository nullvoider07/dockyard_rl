from __future__ import annotations

from dockyard_rl.rewards.diff_utils import parse_diff_paths
from dockyard_rl.rewards.interfaces import RewardFunction, RewardVerificationResult


class IntegrityReward(RewardFunction):
    """Gate an inner reward to zero when the agent patch edits held-out tests.

    The held-out test files are the paths the gold ``test_patch`` touches. The
    executor's force-restore already makes those files canonical before tests
    run, so this is the explicit penalty for an agent patch that attempts to
    modify them — it cannot influence the verdict, and it scores zero.

    Defends against the test-tampering vectors that survive the /task/submit
    model: editing existing test files, replacing their contents, or adding a
    file at a held-out test path. Detection is a static comparison of the two
    diffs; no executor call is made here.

    Args:
        inner_reward: The reward whose result is returned when the agent patch
            leaves the held-out test files untouched.
    """

    def __init__(self, inner_reward: RewardFunction) -> None:
        self._inner = inner_reward

    def __call__(self, trajectory: dict) -> RewardVerificationResult:
        held_out = parse_diff_paths(trajectory.get("test_patch", ""))
        agent = parse_diff_paths(trajectory.get("patch", ""))
        touched = sorted(agent & held_out)
        if touched:
            return RewardVerificationResult(
                reward=0.0,
                status="integrity_failure",
                evidence_hash=None,
                failure_reason=f"patch modifies held-out test file(s): {touched}",
            )
        return self._inner(trajectory)
