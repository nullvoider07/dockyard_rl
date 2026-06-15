"""HLE (Humanity's Last Exam) answer grading: LLM-judge with exact-match fallback.

HLE prompts every question with the fixed format "Explanation: … / Answer: … /
Confidence: …%". Official grading uses an LLM judge that extracts the final
answer and decides correctness semantically. This module replicates that judge
(OpenAI-compatible Chat Completions, configured by environment variables) and
falls back to a deterministic normalized exact-match on the extracted ``Answer:``
line whenever no judge endpoint is configured or a judge call fails — so grading
always runs, and is judge-graded whenever a judge is available.

Judge configuration (all optional; judge disabled unless base_url + model set):
    DOCKYARD_HLE_JUDGE_BASE_URL   e.g. https://api.openai.com/v1
    DOCKYARD_HLE_JUDGE_API_KEY    bearer token (falls back to OPENAI_API_KEY)
    DOCKYARD_HLE_JUDGE_MODEL      e.g. gpt-4o-mini, o3-mini
    DOCKYARD_HLE_JUDGE_TIMEOUT    per-call seconds (default 60)
"""

from __future__ import annotations

import os
import re
import string
from typing import Optional

import requests

# Official HLE judge prompt (centerforaisafety/hle).
_JUDGE_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based on \
the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. \
Put the extracted answer as 'None' if there is no exact, final answer to extract \
from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based \
on [correct_answer], focusing only on if there are meaningful differences between \
[correct_answer] and the extracted_final_answer. Do not comment on any background \
to the problem, do not attempt to solve the problem, do not argue for any answer \
different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given \
above, or is within a small margin of error for numerical problems. Answer 'no' \
otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if \
the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. \
Put 100 if there is no confidence score available."""

_ANSWER_RE = re.compile(r"(?is)answer\s*:\s*(.+?)(?:\n|$)")
_CORRECT_RE = re.compile(r"(?im)^\s*correct\s*:\s*(yes|no)\b")


def extract_answer(response: str) -> str:
    """Extract the model's final answer from the official HLE response format.

    Returns the content of the last ``Answer:`` line; if none is present, the
    last non-empty line; if the response is empty, an empty string.
    """
    if not response:
        return ""
    matches = _ANSWER_RE.findall(response)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    return lines[-1] if lines else response.strip()


def normalize(text: str) -> str:
    """Normalize for exact-match: casefold, strip surrounding punctuation/quotes,
    collapse internal whitespace, and drop a trailing period."""
    s = (text or "").strip().casefold()
    s = s.strip(string.punctuation + string.whitespace + "“”‘’\"'")
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(response: str, gold: str) -> bool:
    """Normalized exact-match of the extracted answer against the gold answer."""
    return normalize(extract_answer(response)) == normalize(gold)


class HLEJudgeClient:
    """OpenAI-compatible Chat Completions judge for HLE answers.

    ``enabled`` is False unless both a base URL and a model are configured, in
    which case callers fall back to exact-match. ``judge`` returns True/False, or
    None when the judge could not be reached or its verdict could not be parsed
    (callers fall back to exact-match on None).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = (base_url if base_url is not None
                         else os.environ.get("DOCKYARD_HLE_JUDGE_BASE_URL", "")).rstrip("/")
        self.api_key = (api_key if api_key is not None
                        else os.environ.get("DOCKYARD_HLE_JUDGE_API_KEY")
                        or os.environ.get("OPENAI_API_KEY", ""))
        self.model = model if model is not None else os.environ.get("DOCKYARD_HLE_JUDGE_MODEL", "")
        self.timeout = float(
            timeout if timeout is not None
            else os.environ.get("DOCKYARD_HLE_JUDGE_TIMEOUT", "60")
        )
        self._session = session or requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def chat(self, messages: list[dict], *, temperature: float = 0) -> Optional[str]:
        """Raw OpenAI-compatible chat completion; returns content or None on any
        transport/parse failure. Shared by HLE and GDPval grading."""
        if not self.enabled:
            return None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        try:
            resp = self._session.post(
                f"{self.base_url}/chat/completions",
                json=body, headers=headers, timeout=self.timeout,
            )
            if resp.status_code != 200:
                return None
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.exceptions.RequestException, KeyError, ValueError, IndexError):
            return None

    def judge(self, question: str, response: str, gold: str) -> Optional[bool]:
        if not self.enabled:
            return None
        content = self.chat([{
            "role": "user",
            "content": _JUDGE_TEMPLATE.format(
                question=question, response=response, correct_answer=gold
            ),
        }])
        if content is None:
            return None
        m = _CORRECT_RE.search(content or "")
        if not m:
            return None
        return m.group(1).lower() == "yes"


def grade(
    response: str,
    gold: str,
    question: str = "",
    *,
    judge: Optional[HLEJudgeClient] = None,
) -> tuple[float, Optional[str]]:
    """Grade one HLE response. Returns (score in {0.0, 1.0}, extracted answer).

    Uses the judge when one is enabled and reachable; otherwise (or on judge
    failure) falls back to normalized exact-match.
    """
    extracted = extract_answer(response)
    if judge is not None and judge.enabled:
        verdict = judge.judge(question, response, gold)
        if verdict is not None:
            return (1.0 if verdict else 0.0, extracted)
    return (1.0 if normalize(extracted) == normalize(gold) else 0.0, extracted)
