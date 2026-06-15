"""RLAIF / Constitutional preference-data generation (D6).

Builds preference pairs without human labels, targeting security/harmlessness and
truth-seeking:

  - Constitutional (critique→revise): for a prompt+response, critique the response
    against each constitution principle, then revise it. The revised answer is the
    chosen sample, the original is rejected — the standard Constitutional-AI route
    to harmless/honest behaviour.
  - Abstention: for prompts the model would hallucinate on, prefer a calibrated
    "I don't know" over a confident wrong answer (truth-seeking / calibration).
  - Factuality: score candidates with a factuality judge and pair best-vs-worst —
    this reuses the D5 online machinery with a factuality scoring template.

Output is ``(chosen, rejected)`` :class:`Candidate` pairs that flow through
``pairs_to_preference_data`` → ``preference_collate_fn`` → the existing DPO step,
exactly like the offline/online paths.

CPU-testable core (this module): the constitution, the critique→revise pipeline,
and the pair constructors — all driven by an injected chat LLM and tokenizer.
At-scale generation against a live model/judge is GPU-deferred (HV).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch

from dockyard_rl.algorithms.online_dpo import Candidate, pairs_to_preference_data

__all__ = [
    "ChatLLM",
    "ConstitutionPrinciple",
    "Constitution",
    "DEFAULT_CONSTITUTION",
    "FACTUALITY_SCORE_TEMPLATE",
    "DEFAULT_ABSTENTION_TEXT",
    "critique",
    "revise",
    "generate_constitutional_revision",
    "build_constitutional_pair",
    "build_abstention_pair",
    "build_rlaif_preference_data",
]


class ChatLLM(Protocol):
    """Minimal chat interface (satisfied by HLEJudgeClient and the judge clients)."""

    enabled: bool

    def chat(self, messages: list[dict], *, temperature: float = 0) -> Optional[str]: ...


@dataclass
class ConstitutionPrinciple:
    """One constitution rule: how to critique against it and how to revise."""

    name: str
    critique_request: str
    revision_request: str


@dataclass
class Constitution:
    principles: list[ConstitutionPrinciple]

    def __post_init__(self) -> None:
        if not self.principles:
            raise ValueError("Constitution must have at least one principle")


DEFAULT_CONSTITUTION = Constitution(
    [
        ConstitutionPrinciple(
            name="harmlessness",
            critique_request=(
                "Identify specific ways the response is harmful, unsafe, unethical, "
                "or could enable wrongdoing."
            ),
            revision_request=(
                "Rewrite the response to remove harmful or unsafe content while "
                "remaining as helpful as is safe."
            ),
        ),
        ConstitutionPrinciple(
            name="honesty",
            critique_request=(
                "Identify any dishonest, misleading, fabricated, or unfounded claims "
                "in the response."
            ),
            revision_request=(
                "Rewrite the response to be truthful and to clearly acknowledge "
                "uncertainty or missing information where appropriate."
            ),
        ),
        ConstitutionPrinciple(
            name="calibration",
            critique_request=(
                "Identify places where the response is overconfident or asserts facts "
                "it cannot support."
            ),
            revision_request=(
                "Rewrite the response to express appropriate calibrated confidence, "
                "abstaining when the answer is not known."
            ),
        ),
    ]
)

# Factuality scoring template for use with online_dpo.PreferenceJudge.
FACTUALITY_SCORE_TEMPLATE = (
    "Assess the factual accuracy of the response to the prompt.\n\n"
    "Prompt:\n{prompt}\n\nResponse:\n{response}\n\n"
    "Rate factual accuracy from 0 to 100, where 100 means fully accurate and "
    "well-calibrated (abstaining when uncertain) and 0 means confidently wrong. "
    "Reply with exactly one line:\nSCORE: <number>"
)

DEFAULT_ABSTENTION_TEXT = (
    "I'm not certain enough to answer this accurately, so I'd rather not guess."
)

_CRITIQUE_TEMPLATE = (
    "Prompt:\n{prompt}\n\nResponse:\n{response}\n\n"
    "Critique request: {critique_request}\n"
    "If the response is already fully acceptable on this point, reply exactly: NO REVISION NEEDED.\n"
    "Critique:"
)
_REVISION_TEMPLATE = (
    "Prompt:\n{prompt}\n\nOriginal response:\n{response}\n\n"
    "Critique:\n{critique}\n\n"
    "Revision request: {revision_request}\nRevised response:"
)
_NO_REVISION_MARKER = "NO REVISION NEEDED"


def _encode(tokenizer: Any, text: str) -> torch.Tensor:
    ids = tokenizer.encode(text)
    if isinstance(ids, torch.Tensor):
        return ids
    return torch.tensor(list(ids), dtype=torch.long)


def critique(
    llm: ChatLLM,
    prompt: str,
    response: str,
    principle: ConstitutionPrinciple,
    *,
    temperature: float = 0.0,
) -> Optional[str]:
    """Critique ``response`` against ``principle``; None on judge failure."""
    return llm.chat(
        [{"role": "user", "content": _CRITIQUE_TEMPLATE.format(
            prompt=prompt, response=response, critique_request=principle.critique_request,
        )}],
        temperature=temperature,
    )


def revise(
    llm: ChatLLM,
    prompt: str,
    response: str,
    critique_text: str,
    principle: ConstitutionPrinciple,
    *,
    temperature: float = 0.0,
) -> Optional[str]:
    """Revise ``response`` given a critique; None on judge failure."""
    return llm.chat(
        [{"role": "user", "content": _REVISION_TEMPLATE.format(
            prompt=prompt, response=response, critique=critique_text,
            revision_request=principle.revision_request,
        )}],
        temperature=temperature,
    )


def generate_constitutional_revision(
    llm: ChatLLM,
    prompt: str,
    response: str,
    constitution: Constitution = DEFAULT_CONSTITUTION,
    *,
    max_principles: Optional[int] = None,
    temperature: float = 0.0,
) -> tuple[str, list[dict[str, str]]]:
    """Iterated critique→revise across the constitution's principles.

    Returns the final revised response and a trace of applied revisions. A
    principle is skipped when the critique returns the no-revision marker or the
    LLM call fails; the running response is threaded through each principle.
    """
    current = response
    trace: list[dict[str, str]] = []
    principles = (
        constitution.principles[:max_principles]
        if max_principles is not None
        else constitution.principles
    )
    for p in principles:
        crit = critique(llm, prompt, current, p, temperature=temperature)
        if crit is None or _NO_REVISION_MARKER in crit:
            continue
        revised = revise(llm, prompt, current, crit, p, temperature=temperature)
        if revised is None or not revised.strip():
            continue
        trace.append({"principle": p.name, "critique": crit, "revision": revised})
        current = revised
    return current, trace


def build_constitutional_pair(
    prompt_text: str,
    original_response: str,
    llm: ChatLLM,
    tokenizer: Any,
    constitution: Constitution = DEFAULT_CONSTITUTION,
    *,
    max_principles: Optional[int] = None,
    temperature: float = 0.0,
) -> Optional[tuple[Candidate, Candidate]]:
    """Build a (revised-chosen, original-rejected) pair, or None if no revision occurred."""
    revised, trace = generate_constitutional_revision(
        llm, prompt_text, original_response, constitution,
        max_principles=max_principles, temperature=temperature,
    )
    if not trace or revised.strip() == original_response.strip():
        return None
    prompt_ids = _encode(tokenizer, prompt_text)
    chosen = Candidate(
        prompt_token_ids=prompt_ids,
        response_token_ids=_encode(tokenizer, revised),
        prompt_text=prompt_text,
        response_text=revised,
    )
    rejected = Candidate(
        prompt_token_ids=prompt_ids,
        response_token_ids=_encode(tokenizer, original_response),
        prompt_text=prompt_text,
        response_text=original_response,
    )
    return chosen, rejected


def build_abstention_pair(
    prompt_text: str,
    hallucinated_response: str,
    tokenizer: Any,
    *,
    abstention_text: str = DEFAULT_ABSTENTION_TEXT,
) -> tuple[Candidate, Candidate]:
    """Prefer a calibrated abstention over a confident wrong answer (truth-seeking).

    Use for prompts the model hallucinates on: chosen = abstention, rejected =
    confident (wrong) response.
    """
    prompt_ids = _encode(tokenizer, prompt_text)
    chosen = Candidate(
        prompt_token_ids=prompt_ids,
        response_token_ids=_encode(tokenizer, abstention_text),
        prompt_text=prompt_text,
        response_text=abstention_text,
    )
    rejected = Candidate(
        prompt_token_ids=prompt_ids,
        response_token_ids=_encode(tokenizer, hallucinated_response),
        prompt_text=prompt_text,
        response_text=hallucinated_response,
    )
    return chosen, rejected


def build_rlaif_preference_data(
    prompt_response_pairs: list[tuple[str, str]],
    llm: ChatLLM,
    tokenizer: Any,
    constitution: Constitution = DEFAULT_CONSTITUTION,
    *,
    max_principles: Optional[int] = None,
    temperature: float = 0.0,
    start_idx: int = 0,
) -> tuple[list, dict[str, int]]:
    """Run constitutional revision over (prompt, response) inputs → preference data.

    Returns ``(preference_data, metrics)`` where ``preference_data`` is a list of
    PreferenceDatumSpec ready for ``preference_collate_fn``, and metrics report how
    many inputs produced a revision.
    """
    pairs: list[tuple[Candidate, Candidate]] = []
    n_no_revision = 0
    for prompt_text, response in prompt_response_pairs:
        pair = build_constitutional_pair(
            prompt_text, response, llm, tokenizer, constitution,
            max_principles=max_principles, temperature=temperature,
        )
        if pair is None:
            n_no_revision += 1
            continue
        pairs.append(pair)
    data = pairs_to_preference_data(pairs, start_idx=start_idx)
    metrics = {
        "num_inputs": len(prompt_response_pairs),
        "num_pairs": len(pairs),
        "num_no_revision": n_no_revision,
    }
    return data, metrics
