from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from typing import Optional

from dockyard_rl.rewards.interfaces import RewardFunction, RewardVerificationResult
from dockyard_rl.sandbox import TaskExecutorError, TaskSpec, run_task

# Test statuses emitted by the per-instance parser.py.
_PASSED = "PASSED"


def _as_list(value: object) -> list[str]:
    """Normalise a fail/pass/selected-files field into a list of strings.

    SWE-bench Pro stores these either as native lists or as Python-repr strings
    of lists (single quotes, embedded apostrophes), so ``ast.literal_eval`` is
    tried before ``json.loads``; a bare unparseable string is treated as one
    element.
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


def _build_entryscript(
    base_commit: str, before_repo_last_line: str, selected_files: list[str]
) -> str:
    """Construct the in-container entryscript (test execution only).

    Mirrors the upstream SWE-bench Pro ``create_entryscript`` sequence: reset/
    checkout the base, apply the agent patch, then run the LAST line of
    ``before_repo_set_cmd`` (which checks out the held-out gold test files from
    the solution commit on top of the patched tree), then run the instance's
    run_script.sh over the comma-joined selected test files.

    The harness is read from the read-only /harness mount and raw logs are
    written to the writable /out mount. The gold ``parser.py`` is NOT run here —
    it runs host-side over the returned logs, fully outside the agent-controlled
    container, so a malicious patch cannot tamper with the verdict. The final
    chmod makes the root-written logs readable back by the executor user.
    ``env_cmds`` is omitted because the image already carries those ENV values.
    """
    selected_csv = ",".join(selected_files)
    return (
        "\ncd /app\n"
        f"git reset --hard {base_commit}\n"
        f"git checkout {base_commit}\n"
        "git apply -v /harness/patch.diff\n"
        f"{before_repo_last_line}\n"
        f"bash /harness/run_script.sh {selected_csv}"
        " > /out/stdout.log 2> /out/stderr.log\n"
        "chmod -R a+rwX /out 2>/dev/null || true\n"
    )


class _ParserError(RuntimeError):
    """Raised when the host-side parser cannot produce a result."""


def _run_parser_hostside(
    parser_script: str, stdout_text: str, stderr_text: str, timeout: int = 120
) -> dict:
    """Run the gold per-instance parser.py over the returned logs, host-side.

    The parser is trusted (pinned benchmark harness, stdlib-only) and never
    enters the agent-controlled container, so it cannot be tampered with. It is
    invoked exactly as upstream does — ``python parser.py <stdout> <stderr>
    <output.json>`` — using ``sys.executable`` (no venv, per project policy).
    """
    with tempfile.TemporaryDirectory(prefix="swebp_parse_") as td:
        parser_path = os.path.join(td, "parser.py")
        stdout_path = os.path.join(td, "stdout.log")
        stderr_path = os.path.join(td, "stderr.log")
        output_path = os.path.join(td, "output.json")
        with open(parser_path, "w", encoding="utf-8") as fh:
            fh.write(parser_script)
        with open(stdout_path, "w", encoding="utf-8") as fh:
            fh.write(stdout_text or "")
        with open(stderr_path, "w", encoding="utf-8") as fh:
            fh.write(stderr_text or "")
        try:
            proc = subprocess.run(
                [sys.executable, parser_path, stdout_path, stderr_path, output_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise _ParserError(f"parser timed out after {timeout}s") from exc
        try:
            with open(output_path, encoding="utf-8") as fh:
                parsed = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise _ParserError(
                f"parser produced no valid output.json (rc={proc.returncode}): "
                + (proc.stderr or "")[:300]
            ) from exc
        if not isinstance(parsed, dict):
            raise _ParserError("parser output.json is not a JSON object")
        return parsed


def _evidence_hash(text: str) -> Optional[str]:
    return hashlib.sha256(text.encode()).hexdigest() if text else None


class SWEBenchProReward(RewardFunction):
    """Score a SWE-bench Pro solution inside its prebuilt per-instance image.

    The agent patch is applied in the instance image, the held-out gold tests
    are checked out (via the last line of ``before_repo_set_cmd``), the
    instance's run_script.sh runs the selected test files, and its parser.py
    emits per-test ``{name, status}`` results. A task is resolved iff every
    FAIL_TO_PASS test and every PASS_TO_PASS test reports ``PASSED``.

    Reads from the trajectory: ``sandbox_url``, ``image`` (full ref),
    ``base_commit``, ``patch`` (agent diff), ``before_repo_set_cmd``,
    ``selected_test_files``, ``run_script`` + ``parser_script`` (per-instance
    harness, embedded by the dataset loader), ``fail_to_pass``, ``pass_to_pass``,
    and optionally ``timeout`` / ``block_network``.

    Args:
        reward_mode: "binary" → 1.0 when resolved else 0.0;
                     "test_pass_rate" → resolved_names / total, clamped [0, 1].
        api_token:   Bearer token for the executor; defaults to the
                     DOCKYARD_SANDBOX_API_TOKEN environment variable.
        block_network: Run the container with ``--network none``. Default False
                     because the per-instance run_script.sh may install packages
                     (e.g. ``npm install``) at test time.
        pull_timeout: Seconds allowed for a cold image pull.
    """

    def __init__(
        self,
        reward_mode: str = "binary",
        api_token: Optional[str] = None,
        block_network: bool = False,
        pull_timeout: int = 1800,
    ) -> None:
        if reward_mode not in ("binary", "test_pass_rate"):
            raise ValueError(
                f"Unsupported reward_mode '{reward_mode}'. "
                "Use 'binary' or 'test_pass_rate'."
            )
        self._reward_mode = reward_mode
        self._api_token = api_token
        self._block_network = block_network
        self._pull_timeout = pull_timeout

    def _error(self, reason: str) -> RewardVerificationResult:
        return RewardVerificationResult(
            reward=0.0,
            status="execution_error",
            evidence_hash=None,
            failure_reason=reason,
        )

    def __call__(self, trajectory: dict) -> RewardVerificationResult:
        image = trajectory.get("image")
        if not image:
            return self._error("trajectory has no image reference")

        fail_to_pass = _as_list(trajectory.get("fail_to_pass"))
        pass_to_pass = _as_list(trajectory.get("pass_to_pass"))
        if not fail_to_pass:
            return self._error("trajectory has no fail_to_pass tests")

        run_script = trajectory.get("run_script") or ""
        parser_script = trajectory.get("parser_script") or ""
        if not run_script or not parser_script:
            return self._error("trajectory missing per-instance run_script/parser")

        selected = _as_list(trajectory.get("selected_test_files"))
        before = (trajectory.get("before_repo_set_cmd") or "").strip()
        before_last = before.split("\n")[-1] if before else ""
        entry = _build_entryscript(
            trajectory.get("base_commit", "HEAD"), before_last, selected
        )

        spec: TaskSpec = {
            "mode":  "image",
            "image": str(image),
            "harness_files": {
                "patch.diff":     trajectory.get("patch", ""),
                "run_script.sh":  run_script,
                "entryscript.sh": entry,
            },
            "entry_command": "bash /harness/entryscript.sh",
            "result_files":  ["stdout.log", "stderr.log"],
            "timeout":       int(trajectory.get("timeout", 1800)),
            "block_network": bool(trajectory.get("block_network", self._block_network)),
            "pull_timeout":  self._pull_timeout,
        }

        try:
            result = run_task(
                trajectory["sandbox_url"], spec, api_token=self._api_token
            )
        except TaskExecutorError as exc:
            return self._error(str(exc))

        if result["status"] != "completed":
            return self._error(
                (result.get("stderr") or "image task did not complete")[:512]
            )

        result_files = result.get("result_files") or {}
        stdout_log = result_files.get("stdout.log")
        stderr_log = result_files.get("stderr.log")
        if stdout_log is None and stderr_log is None:
            return self._error(
                "image task produced no run logs: "
                + (result.get("stderr") or "")[:400]
            )

        # Parse host-side: the gold parser never enters the agent container.
        try:
            parsed = _run_parser_hostside(parser_script, stdout_log or "", stderr_log or "")
        except _ParserError as exc:
            return self._error(str(exc))
        tests = parsed.get("tests", [])

        # Build the set of test names that PASSED. A name appearing both passed
        # and not-passed is treated as not passed.
        passed: set[str] = set()
        not_passed: set[str] = set()
        for t in tests:
            if not isinstance(t, dict):
                continue
            name = str(t.get("name", "")).strip()
            if not name:
                continue
            if str(t.get("status", "")).upper() == _PASSED:
                passed.add(name)
            else:
                not_passed.add(name)
        passed -= not_passed

        evidence = _evidence_hash((stdout_log or "") + (stderr_log or ""))

        required = [n.strip() for n in (fail_to_pass + pass_to_pass)]
        resolved_names = [n for n in required if n in passed]
        resolved = len(resolved_names) == len(required) and len(required) > 0

        if self._reward_mode == "binary":
            reward = 1.0 if resolved else 0.0
        else:
            reward = max(0.0, min(1.0, len(resolved_names) / len(required)))

        failure_reason = (
            None
            if resolved
            else (
                f"unresolved: {len(resolved_names)}/{len(required)} required tests passed "
                f"(fail_to_pass={len(fail_to_pass)}, pass_to_pass={len(pass_to_pass)})"
            )
        )
        return RewardVerificationResult(
            reward=reward,
            status="ok",
            evidence_hash=evidence,
            failure_reason=failure_reason,
        )
