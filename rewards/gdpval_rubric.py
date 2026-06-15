"""GDPval rubric grading: score a deliverable against the task's rubric via an LLM judge.

Each GDPval task ships a ``rubric_json`` — a list of weighted criteria
``{"score": <weight>, "criterion": <text>, "required": <null|bool>,
"rubric_item_id": <id>, ...}``. This module asks an LLM judge (OpenAI-compatible,
reusing ``HLEJudgeClient.chat``) to mark each criterion satisfied/not against the
model's deliverable, then computes a weighted fraction:

    reward = sum(weight for met criteria) / sum(positive weights), clamped [0, 1]

with negative-weight criteria acting as penalties and any unmet ``required``
criterion gating the reward to 0.

Unlike HLE there is NO deterministic fallback — rubric grading is judgmental, so a
judge endpoint is REQUIRED. Configure it via (GDPval-specific, else shared HLE):
    DOCKYARD_GDPVAL_JUDGE_BASE_URL  (else DOCKYARD_HLE_JUDGE_BASE_URL)
    DOCKYARD_GDPVAL_JUDGE_MODEL     (else DOCKYARD_HLE_JUDGE_MODEL)
    DOCKYARD_GDPVAL_JUDGE_API_KEY   (else DOCKYARD_HLE_JUDGE_API_KEY / OPENAI_API_KEY)
    DOCKYARD_GDPVAL_JUDGE_TIMEOUT   (default 120)

Limitation: a text deliverable cannot satisfy file/format criteria (e.g. "the
workbook contains a worksheet named X"); those criteria score 0 against text
output. The faithful path is the agentic file-producing environment (deferred).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from typing import NamedTuple, Optional

from dockyard_rl.rewards.hle_grader import HLEJudgeClient
from dockyard_rl.rewards.interfaces import RewardFunction, RewardVerificationResult

_GRADE_TEMPLATE = """\
You are grading a work deliverable against a rubric. Be strict and literal.

[Task]:
{prompt}

[Deliverable]:
{deliverable}

[Rubric] — for each criterion below, decide whether the [Deliverable] satisfies \
it (true) or not (false). If the deliverable cannot demonstrate a criterion \
(e.g. it requires a file artifact that is not present), mark it false.

{criteria}

Respond with ONLY a JSON object mapping each criterion id to true or false, e.g.
{{"<id>": true, "<id>": false}}. Output no other text."""


class RubricItem(NamedTuple):
    item_id: str
    criterion: str
    weight: float
    required: bool


def parse_rubric(rubric_json: object) -> list[RubricItem]:
    """Parse a GDPval ``rubric_json`` (JSON/repr string or list) into RubricItems."""
    data: object = rubric_json
    if isinstance(rubric_json, str):
        s = rubric_json.strip()
        if not s:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                data = parser(s)
                break
            except (ValueError, SyntaxError):
                data = None
        if data is None:
            return []
    if not isinstance(data, list):
        return []
    items: list[RubricItem] = []
    for i, raw in enumerate(data):
        if not isinstance(raw, dict):
            continue
        criterion = str(raw.get("criterion", "")).strip()
        if not criterion:
            continue
        item_id = str(raw.get("rubric_item_id") or f"item_{i}")
        try:
            weight = float(raw.get("score", 1) or 0)
        except (TypeError, ValueError):
            weight = 1.0
        items.append(
            RubricItem(item_id, criterion, weight, bool(raw.get("required")))
        )
    return items


def resolve_judge(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> HLEJudgeClient:
    """Build a judge client from GDPval config, falling back to the shared HLE
    judge env vars then OPENAI_API_KEY."""
    env = os.environ.get
    return HLEJudgeClient(
        base_url=base_url if base_url is not None
        else (env("DOCKYARD_GDPVAL_JUDGE_BASE_URL") or env("DOCKYARD_HLE_JUDGE_BASE_URL", "")),
        api_key=api_key if api_key is not None
        else (env("DOCKYARD_GDPVAL_JUDGE_API_KEY") or env("DOCKYARD_HLE_JUDGE_API_KEY")
              or env("OPENAI_API_KEY", "")),
        model=model if model is not None
        else (env("DOCKYARD_GDPVAL_JUDGE_MODEL") or env("DOCKYARD_HLE_JUDGE_MODEL", "")),
        timeout=timeout if timeout is not None
        else float(env("DOCKYARD_GDPVAL_JUDGE_TIMEOUT", "120")),
    )


def _parse_verdicts(content: str, item_ids: list[str]) -> Optional[dict[str, bool]]:
    """Extract a {id: bool} verdict map from the judge's JSON response."""
    if not content:
        return None
    text = content.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    valid = set(item_ids)
    return {k: bool(v) for k, v in obj.items() if k in valid}


def _evidence_hash(text: str) -> Optional[str]:
    return hashlib.sha256(text.encode()).hexdigest() if text else None


class GDPvalRubricReward(RewardFunction):
    """Score a GDPval deliverable against ``rubric_json`` with an LLM judge.

    Reads from the trajectory: ``deliverable`` (the model's text output),
    ``prompt`` (the task), ``rubric_json`` (the task rubric). Returns a weighted
    fraction of satisfied criteria (or, in "binary" mode, 1.0 iff all positive
    criteria are met). Requires a configured judge; without one, every sample is
    an ``execution_error`` (there is no deterministic rubric fallback).

    Args:
        reward_mode: "weighted" (default) → weighted fraction; "binary" → 1.0 iff
                     fraction == 1.0 and no required criterion is unmet.
        base_url/api_key/model/timeout: judge overrides (else env vars).
    """

    def __init__(
        self,
        reward_mode: str = "weighted",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        if reward_mode not in ("weighted", "binary"):
            raise ValueError(
                f"Unsupported reward_mode '{reward_mode}'. Use 'weighted' or 'binary'."
            )
        self._reward_mode = reward_mode
        self._judge = resolve_judge(base_url, api_key, model, timeout)

    def _error(self, reason: str) -> RewardVerificationResult:
        return RewardVerificationResult(
            reward=0.0, status="execution_error", evidence_hash=None,
            failure_reason=reason,
        )

    def __call__(self, trajectory: dict) -> RewardVerificationResult:
        if not self._judge.enabled:
            return self._error(
                "no GDPval judge configured (set DOCKYARD_GDPVAL_JUDGE_BASE_URL + "
                "DOCKYARD_GDPVAL_JUDGE_MODEL); rubric grading has no deterministic fallback"
            )
        items = parse_rubric(trajectory.get("rubric_json"))
        if not items:
            return self._error("trajectory has no parseable rubric")
        deliverable = trajectory.get("deliverable") or ""
        if not deliverable.strip():
            return self._error("empty deliverable")

        criteria_block = "\n".join(
            f"- id: {it.item_id}\n  criterion: {it.criterion}" for it in items
        )
        content = self._judge.chat([{
            "role": "user",
            "content": _GRADE_TEMPLATE.format(
                prompt=trajectory.get("prompt", ""),
                deliverable=deliverable,
                criteria=criteria_block,
            ),
        }])
        if content is None:
            return self._error("judge call failed")
        verdicts = _parse_verdicts(content, [it.item_id for it in items])
        if verdicts is None:
            return self._error("could not parse judge verdicts")

        total_positive = sum(it.weight for it in items if it.weight > 0)
        achieved = sum(it.weight for it in items if verdicts.get(it.item_id))
        required_unmet = any(
            it.required and not verdicts.get(it.item_id) for it in items
        )

        if total_positive <= 0:
            return self._error("rubric has no positive-weight criteria")
        if required_unmet:
            fraction = 0.0
        else:
            fraction = max(0.0, min(1.0, achieved / total_positive))

        reward = (1.0 if fraction >= 1.0 else 0.0) if self._reward_mode == "binary" else fraction
        n_met = sum(1 for it in items if verdicts.get(it.item_id))
        failure_reason = (
            None if reward >= 1.0
            else f"rubric: {n_met}/{len(items)} criteria met, weighted {achieved:.1f}/{total_positive:.1f}"
            + (" (required criterion unmet)" if required_unmet else "")
        )
        return RewardVerificationResult(
            reward=reward,
            status="ok",
            evidence_hash=_evidence_hash(content),
            failure_reason=failure_reason,
        )
