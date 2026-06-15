"""Answer parsing for multiple-choice evaluation verifiers.

Pure string normalization + answer-marker constants consumed by the math
environment's ``english_multichoice`` / ``multilingual_multichoice`` verifier
workers (``environments/math_environment.py``) to score MMLU / GPQA-style
multiple-choice benchmarks. No GPU/cluster dependency — unit-testable on CPU.

The English verifier uses its own ``Answer:\\s*([A-Z])`` pattern plus
:func:`normalize_response` / :func:`normalize_extracted_answer`. The multilingual
verifier additionally scans :data:`MULTILINGUAL_ANSWER_REGEXES` (the answer marker
written across languages) with :data:`MULTILINGUAL_ANSWER_PATTERN_TEMPLATE`. The
marker list follows the standard multiple-choice eval harness; extend it if a new
language is needed.
"""

from __future__ import annotations

# "Answer:" (and equivalents) across languages — scanned by the multilingual
# verifier in order; the first match wins.
MULTILINGUAL_ANSWER_REGEXES: list[str] = [
    r"Answer\s*:",
    r"Réponse\s*:",
    r"Antwort\s*:",
    r"Respuesta\s*:",
    r"Risposta\s*:",
    r"Resposta\s*:",
    r"Jawaban\s*:",
    r"Trả lời\s*:",
    r"答案\s*：",
    r"答案\s*:",
    r"答\s*：",
    r"答\s*:",
    r"정답\s*:",
    r"답변\s*:",
    r"답\s*:",
    r"उत्तर\s*:",
    r"উত্তর\s*:",
    r"উত্তরঃ",
    r"الإجابة\s*:",
    r"الجواب\s*:",
    r"إجابة\s*:",
    r"الإجابة الصحيحة\s*:",
    r"الإجابة هي\s*:",
]

# Filled with one regex above; captures the chosen option marker (Latin, Arabic,
# Bengali, or fullwidth). Case-insensitive.
MULTILINGUAL_ANSWER_PATTERN_TEMPLATE: str = r"(?i){}\s*([A-D]|[أ-د]|[অ-ঘ]|[Ａ-Ｄ])"


def normalize_response(response: str) -> str:
    """Strip markdown / LaTeX wrappers that would otherwise block an answer match."""
    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


def normalize_extracted_answer(extracted_answer: str) -> str:
    """Map a captured option marker (Latin / Arabic / Bengali / fullwidth) to A-D.

    The captured group may be a lowercase Latin letter (the English pattern is
    case-insensitive) or a non-Latin script letter; this returns a single
    uppercase ``A``-``D`` for an exact comparison against the gold label.
    """
    return (
        extracted_answer
        # Arabic
        .replace("أ", "A").replace("ب", "B").replace("ج", "C").replace("د", "D")
        # Bengali
        .replace("অ", "A").replace("ব", "B").replace("ড", "C").replace("ঘ", "D")
        # Fullwidth Latin
        .replace("Ａ", "A").replace("Ｂ", "B").replace("Ｃ", "C").replace("Ｄ", "D")
        .strip()
        .upper()
    )
