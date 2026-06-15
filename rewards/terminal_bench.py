from __future__ import annotations

import hashlib
import json
from typing import Optional

from dockyard_rl.rewards.interfaces import RewardFunction, RewardVerificationResult

# CTRF (Common Test Report Format) status string for a passing test.
_PASSED = "passed"


def _evidence_hash(text: str) -> Optional[str]:
    return hashlib.sha256(text.encode()).hexdigest() if text else None


def _parse_ctrf(ctrf_text: str) -> Optional[dict]:
    """Extract {passed, failed, total} from a CTRF report, or None.

    pytest-json-ctrf writes ``{"results": {"summary": {"tests", "passed",
    "failed", ...}, "tests": [...]}}``. The summary is authoritative for counts;
    if it is absent the per-test list is tallied as a fallback.
    """
    try:
        data = json.loads(ctrf_text)
    except (ValueError, TypeError):
        return None
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, dict):
        return None
    summary = results.get("summary")
    if isinstance(summary, dict) and "passed" in summary:
        passed = int(summary.get("passed", 0))
        failed = int(summary.get("failed", 0))
        total = int(summary.get("tests", passed + failed))
        return {"passed": passed, "failed": failed, "total": total}
    tests = results.get("tests")
    if isinstance(tests, list):
        passed = sum(
            1 for t in tests
            if isinstance(t, dict) and str(t.get("status", "")).lower() == _PASSED
        )
        total = len(tests)
        return {"passed": passed, "failed": total - passed, "total": total}
    return None


class TerminalBenchReward(RewardFunction):
    """Score a Terminal-Bench task from its verifier output (host-side parse).

    Terminal-Bench grades the FINAL container state: after the agent's turns, the
    environment late-injects the task's ``tests/`` and runs ``test.sh``, which
    runs pytest and writes ``/logs/verifier/ctrf.json`` (per-test report) plus
    ``/logs/verifier/reward.txt`` (the binary ``1``/``0`` pytest verdict). This
    reward is a pure parser of those artifacts — it issues no executor call; the
    environment drives the session and hands the results over via the trajectory.

    Integrity is structural: the tests are injected only at finish time (the
    agent never sees them during its turns), and the inject overwrites any file
    the agent may have pre-placed at the tests/logs paths, so there is no
    in-rollout tamper vector. The residual surface (a poisoned in-container
    toolchain influencing pytest) is inherent to any state-inspecting verifier.

    Reads from the trajectory (populated by the environment after /finish):
        ctrf                (str|None) — content of /logs/verifier/ctrf.json
        reward_txt          (str|None) — content of /logs/verifier/reward.txt
        verifier_exit_code  (int|None) — test.sh exit code
        verifier_error      (str|None) — set if /finish raised or timed out

    Args:
        reward_mode: "binary" → 1.0 when the verifier passed else 0.0;
                     "test_pass_rate" → passed/(passed+failed), clamped [0, 1].
    """

    def __init__(self, reward_mode: str = "binary") -> None:
        if reward_mode not in ("binary", "test_pass_rate"):
            raise ValueError(
                f"Unsupported reward_mode '{reward_mode}'. "
                "Use 'binary' or 'test_pass_rate'."
            )
        self._reward_mode = reward_mode

    def _error(self, reason: str) -> RewardVerificationResult:
        return RewardVerificationResult(
            reward=0.0,
            status="execution_error",
            evidence_hash=None,
            failure_reason=reason,
        )

    def __call__(self, trajectory: dict) -> RewardVerificationResult:
        ctrf_text = trajectory.get("ctrf")
        reward_txt = trajectory.get("reward_txt")
        exit_code = trajectory.get("verifier_exit_code")
        verifier_error = trajectory.get("verifier_error")

        counts = _parse_ctrf(ctrf_text) if ctrf_text else None

        # Binary verdict precedence: reward.txt (the pytest exit-code verdict
        # written by the injected harness) → CTRF counts → verifier exit code.
        passed_flag: Optional[bool] = None
        if reward_txt is not None:
            s = reward_txt.strip()
            if s.startswith("1"):
                passed_flag = True
            elif s.startswith("0"):
                passed_flag = False
        if passed_flag is None and counts is not None:
            passed_flag = counts["total"] > 0 and counts["failed"] == 0 and counts["passed"] > 0
        if passed_flag is None and exit_code is not None:
            passed_flag = int(exit_code) == 0

        if passed_flag is None:
            return self._error(
                verifier_error or "verifier produced no parseable result"
            )

        evidence = _evidence_hash((ctrf_text or "") + "\n" + (reward_txt or ""))

        if self._reward_mode == "binary":
            reward = 1.0 if passed_flag else 0.0
        else:
            if counts is not None and (counts["passed"] + counts["failed"]) > 0:
                denom = counts["passed"] + counts["failed"]
                reward = max(0.0, min(1.0, counts["passed"] / denom))
            else:
                reward = 1.0 if passed_flag else 0.0

        if passed_flag:
            failure_reason = None
        elif counts is not None:
            failure_reason = (
                f"verifier failed: {counts['passed']} passed / {counts['failed']} failed "
                f"of {counts['total']} (exit={exit_code})"
            )
        else:
            failure_reason = f"verifier failed (exit={exit_code}); no CTRF report"

        return RewardVerificationResult(
            reward=reward,
            status="ok",
            evidence_hash=evidence,
            failure_reason=failure_reason,
        )
