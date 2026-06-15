from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from typing import Optional

from dockyard_rl.rewards.interfaces import RewardFunction, RewardVerificationResult
from dockyard_rl.sandbox import TaskExecutorError, TaskSpec, run_task

# Clean-rebuild grading entryscript, run inside the per-instance task image.
# Mirrors the official ProgramBench Evaluator: wipe the workspace, extract the
# agent submission, seed a deterministic git repo, run compile.sh to produce
# ./executable, stash+hash it, then for each test branch restore the executable
# and run that branch's eval/run.sh (pytest → eval/results.xml). The held-out
# tests never reached the agent — they are injected only here, in a fresh image,
# so a prebuilt ./executable + stub compile.sh cannot pass (the wipe removes
# them before compile.sh runs). The gold parser/scoring runs host-side.
_GRADING_ENTRYSCRIPT = r"""
set +e
STASH=/tmp/_pb_stashed_executable
cd /workspace
rm -rf /workspace/* /workspace/.[!.]* 2>/dev/null
tar -C /workspace -xzf /harness/submission.tar.gz 2>/dev/null
if [ ! -d .git ]; then
  export GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' GIT_COMMITTER_DATE='2000-01-01T00:00:00Z'
  git -c init.defaultBranch=gold init -q 2>/dev/null
  git -c user.email=gold@local -c user.name=gold -c commit.gpgsign=false add -A 2>/dev/null
  git -c user.email=gold@local -c user.name=gold -c commit.gpgsign=false commit -q --allow-empty -m gold 2>/dev/null
fi
chmod +x ./compile.sh 2>/dev/null
./compile.sh > /out/compile.log 2>&1
echo $? > /out/compile.exit
if [ -f ./executable ]; then
  cp ./executable "$STASH" 2>/dev/null
  sha256sum "$STASH" 2>/dev/null | awk '{print $1}' > /out/executable_hash
fi
for tar in /harness/branch_*.tar.gz; do
  [ -f "$tar" ] || continue
  b=$(basename "$tar" .tar.gz); b=${b#branch_}
  rm -rf /workspace/eval /workspace/results.xml /workspace/eval/results.xml 2>/dev/null
  if [ -f "$STASH" ]; then cp "$STASH" ./executable 2>/dev/null && chmod +x ./executable 2>/dev/null; fi
  tar -C /workspace -xzf "$tar" 2>/dev/null
  [ -f eval/run.sh ] && sed -i 's/--timeout-method=thread/--timeout-method=signal/g' eval/run.sh 2>/dev/null
  chmod +x ./eval/run.sh 2>/dev/null
  ./eval/run.sh > "/out/run_${b}.log" 2>&1
  cp eval/results.xml "/out/results_${b}.xml" 2>/dev/null
done
chmod -R a+rwX /out 2>/dev/null || true
"""


def _evidence_hash(text: str) -> Optional[str]:
    return hashlib.sha256(text.encode()).hexdigest() if text else None


def _passed_names(xml_text: str) -> dict[str, bool]:
    """Map each JUnit testcase ``classname.name`` to passed (no failure/error/skip).

    Mirrors the official parser: status is ``passed`` iff the ``<testcase>`` has
    no ``<failure>``/``<error>``/``<skipped>`` child.
    """
    out: dict[str, bool] = {}
    if not xml_text or not xml_text.strip():
        return out
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for case in root.iter("testcase"):
        cls = case.get("classname") or ""
        nm = case.get("name") or ""
        full = f"{cls}.{nm}" if cls else nm
        if not full:
            continue
        bad = any(
            case.find(tag) is not None for tag in ("failure", "error", "skipped")
        )
        out[full] = not bad
    return out


class ProgramBenchReward(RewardFunction):
    """Score a ProgramBench cleanroom submission by clean-rebuild + behavioural tests.

    The agent's exported workspace (``submission_tar``) is rebuilt from scratch
    in the per-instance image (``image``): wipe → extract → ``compile.sh`` →
    ``./executable``. Each selected test branch's suite (``branches[*].tar``) is
    then run against that rebuilt executable, and the passed pytest node IDs are
    matched against the expected list (``branches[*].expected``) from tests.json.

    Score = resolved / total over the union of selected branches, gated on a
    successful compile (a build failure yields reward 0).

    Reads from the trajectory:
        sandbox_url   (str)
        image         (str)            — programbench/<id _1776_>:task_cleanroom
        submission_tar(str, base64)    — gzip tar of the agent /workspace
        branches      (dict)           — {branch: {"tar_b64": str, "expected": [node_id]}}
        timeout       (int, optional)  — grading container wall-clock cap
        block_network (bool, optional) — default True (cleanroom: offline)

    Args:
        reward_mode: "binary" → 1.0 iff every expected test passes;
                     "test_pass_rate" → resolved/total, clamped [0, 1].
    """

    def __init__(
        self,
        reward_mode: str = "test_pass_rate",
        api_token: Optional[str] = None,
        block_network: bool = True,
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
            reward=0.0, status="execution_error", evidence_hash=None,
            failure_reason=reason,
        )

    def __call__(self, trajectory: dict) -> RewardVerificationResult:
        image = trajectory.get("image")
        if not image:
            return self._error("trajectory has no image reference")
        submission = trajectory.get("submission_tar")
        if not submission:
            return self._error("trajectory has no submission (agent produced nothing)")
        branches = trajectory.get("branches") or {}
        if not branches:
            return self._error("trajectory has no test branches")

        harness_files: dict[str, object] = {
            "submission.tar.gz": {"b64": submission},
            "entryscript.sh": _GRADING_ENTRYSCRIPT,
        }
        expected_by_branch: dict[str, list[str]] = {}
        for name, spec in branches.items():
            tar_b64 = (spec or {}).get("tar_b64")
            if not tar_b64:
                continue
            harness_files[f"branch_{name}.tar.gz"] = {"b64": tar_b64}
            expected_by_branch[name] = [str(t) for t in (spec.get("expected") or [])]
        if not expected_by_branch:
            return self._error("no usable test branches (missing tar/expected)")

        result_files = ["compile.exit", "compile.log", "executable_hash"]
        result_files += [f"results_{b}.xml" for b in expected_by_branch]

        spec: TaskSpec = {
            "mode":          "image",
            "image":         str(image),
            "harness_files": harness_files,
            "entry_command": "bash /harness/entryscript.sh",
            "result_files":  result_files,
            "timeout":       int(trajectory.get("timeout", 3600)),
            "block_network": bool(trajectory.get("block_network", self._block_network)),
            "pull_timeout":  self._pull_timeout,
        }

        try:
            result = run_task(trajectory["sandbox_url"], spec, api_token=self._api_token)
        except TaskExecutorError as exc:
            return self._error(str(exc))
        if result["status"] != "completed":
            return self._error(
                (result.get("stderr") or "grading task did not complete")[:512]
            )

        files = result.get("result_files") or {}
        compile_exit_raw = (files.get("compile.exit") or "").strip()
        try:
            compile_exit = int(compile_exit_raw.split()[0]) if compile_exit_raw else None
        except ValueError:
            compile_exit = None

        if compile_exit is None:
            return self._error(
                "grading produced no compile result: "
                + (files.get("compile.log") or result.get("stderr") or "")[:400]
            )
        if compile_exit != 0:
            log = (files.get("compile.log") or "").strip()
            return RewardVerificationResult(
                reward=0.0,
                status="ok",
                evidence_hash=_evidence_hash(files.get("executable_hash") or compile_exit_raw),
                failure_reason=f"compile.sh failed (exit {compile_exit}): {log[-300:]}",
            )

        resolved = 0
        total = 0
        evidence_src = files.get("executable_hash") or ""
        for branch, expected in expected_by_branch.items():
            xml_text = files.get(f"results_{branch}.xml")
            passed_map = _passed_names(xml_text or "")
            total += len(expected)
            resolved += sum(1 for n in expected if passed_map.get(n, False))
            evidence_src += xml_text or ""

        if total == 0:
            return self._error("no expected tests across selected branches")

        if self._reward_mode == "binary":
            reward = 1.0 if resolved == total else 0.0
        else:
            reward = max(0.0, min(1.0, resolved / total))

        failure_reason = (
            None if resolved == total
            else f"behavioural mismatch: {resolved}/{total} expected tests passed "
                 f"across {len(expected_by_branch)} branch(es)"
        )
        return RewardVerificationResult(
            reward=reward,
            status="ok",
            evidence_hash=_evidence_hash(evidence_src),
            failure_reason=failure_reason,
        )
