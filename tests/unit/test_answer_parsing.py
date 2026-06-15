"""CPU unit tests for the multiple-choice answer parser (dockyard_rl.evals.answer_parsing).

Mirrors the extraction the math env's english_multichoice verifier performs:
normalize the response, regex out ``Answer: <letter>``, normalize the letter, and
compare to the gold label. No GPU/cluster.
"""

from __future__ import annotations

import re

from dockyard_rl.evals.answer_parsing import (
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    normalize_extracted_answer,
    normalize_response,
)

# The exact pattern the english_multichoice verifier uses.
_ENGLISH = re.compile(r"(?i)Answer\s*:[ \t]*([A-Z])")


def _english_score(response: str, gold: str) -> float:
    response = normalize_response(response)
    gold = normalize_response(gold)
    m = _ENGLISH.search(response)
    extracted = normalize_extracted_answer(m.group(1)) if m else None
    return 1.0 if extracted == gold else 0.0


def test_normalize_response_strips_markdown_latex():
    assert normalize_response("**Answer: B**") == "Answer: B"
    assert normalize_response("$\\boxed{C}$") == "C"
    # \text and the opening { are stripped; a bare } may remain (harmless — the
    # verifier extracts the letter after "Answer:", so trailing braces don't matter).
    out = normalize_response("The answer is \\text{D}")
    assert "D" in out and "\\text" not in out and "{" not in out


def test_normalize_extracted_answer_uppercases_and_strips():
    assert normalize_extracted_answer("a") == "A"
    assert normalize_extracted_answer(" b ") == "B"
    # Arabic / Bengali / fullwidth map to Latin A-D.
    assert normalize_extracted_answer("ج") == "C"
    assert normalize_extracted_answer("Ｄ") == "D"


def test_english_scoring_correct_and_incorrect():
    assert _english_score("Reasoning... Answer: B", "B") == 1.0
    assert _english_score("Answer: A", "B") == 0.0
    # Case-insensitive + markdown wrapper.
    assert _english_score("**answer: c**", "C") == 1.0
    # No parsable answer → 0.
    assert _english_score("I think it is the second one.", "B") == 0.0
    # Trailing prose after the marker still extracts the letter.
    assert _english_score("Answer: D. This is correct because...", "D") == 1.0


def test_multilingual_constants_well_formed():
    assert MULTILINGUAL_ANSWER_REGEXES, "expected at least one answer marker"
    assert r"Answer\s*:" in MULTILINGUAL_ANSWER_REGEXES
    # The template + each marker must compile and capture a letter.
    for marker in MULTILINGUAL_ANSWER_REGEXES:
        pat = re.compile(MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(marker))
        assert pat.groups == 1
    m = re.search(
        MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(r"Answer\s*:"), "Answer: C"
    )
    assert m is not None and normalize_extracted_answer(m.group(1)) == "C"
