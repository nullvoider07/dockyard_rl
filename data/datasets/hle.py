"""HLE (Humanity's Last Exam) dataset loader for Project Dockyard.

Trainer-side only. Loads ``cais/hle`` (gated — requires an HF token) and presents
each text question as a ``messages`` + metadata row, scored by the HLE verifier
environment (LLM judge with normalized exact-match fallback). HLE is a 2,500-
question held-out exam: it is intended as a **validation** benchmark (point
``data.validation`` at it), not a training set.

The current generation pipeline is text-only, so the ~14% of questions that carry
an image (``image`` field non-empty) are filtered out by default.

Schema (cais/hle, ``test`` split): id, question, image, answer, answer_type
(exactMatch | multipleChoice), rationale, raw_subject, category, canary.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from datasets import load_dataset
from dockyard_rl.data.datasets.raw_dataset import RawDataset

# Official HLE response-format instruction (centerforaisafety/hle). The verifier
# extracts the "Answer:" line, so the model must produce it.
_SYSTEM_PROMPT = """\
Your response should be in the following format:
Explanation: {your explanation for your answer choice}
Answer: {your chosen answer}
Confidence: {your confidence score between 0% and 100% for your answer}"""


class HLEDataset(RawDataset):
    """HLE questions exposed as ``messages`` + metadata rows.

    Args:
        hf_dataset_name: HuggingFace dataset name or local path.
        split:           Dataset split (default: "test").
        text_only:       Drop questions that carry an image (default: True; the
                         generation pipeline is text-only).
        instance_ids:    Optional subset of question ids.
        shuffle_seed:    If set, shuffle with this seed.
    """

    def __init__(
        self,
        hf_dataset_name: str = "cais/hle",
        split: str = "test",
        text_only: bool = True,
        instance_ids: Optional[list[str]] = None,
        shuffle_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.task_name = "hle"

        print(f"Loading HLE dataset: {hf_dataset_name} (split={split})", flush=True)
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )
        try:
            raw = load_dataset(hf_dataset_name, split=split, token=token)
        except Exception as exc:  # noqa: BLE001 — surface the gating requirement
            raise ValueError(
                f"Failed to load {hf_dataset_name!r} (split={split}). cais/hle is "
                "gated: accept the terms on HuggingFace and export a token as "
                "HF_TOKEN (or HUGGING_FACE_HUB_TOKEN). Underlying error: "
                f"{exc}"
            ) from exc

        if instance_ids:
            id_set = set(instance_ids)
            raw = raw.filter(lambda x: x.get("id") in id_set)

        if text_only:
            n_before = len(raw)
            raw = raw.filter(lambda x: not (x.get("image") or "").strip())
            dropped = n_before - len(raw)
            if dropped:
                print(f"  ⚠ Dropped {dropped} multimodal question(s) (text-only)", flush=True)

        if shuffle_seed is not None:
            raw = raw.shuffle(seed=shuffle_seed)

        self.dataset = raw.map(self.format_data, remove_columns=raw.column_names)
        self.val_dataset = None
        print(f"  ✓ Loaded {len(self.dataset)} HLE questions", flush=True)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        question = (data.get("question") or "").strip()
        answer = (data.get("answer") or "").strip()
        return {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            "task_name":   self.task_name,
            "id":          data.get("id", ""),
            # Held-out scoring inputs routed into extra_env_info by the processor.
            "ground_truth": answer,
            "question":     question,
            "answer_type":  (data.get("answer_type") or "").strip(),
            "category":     (data.get("category") or "").strip(),
            "raw_subject":  (data.get("raw_subject") or "").strip(),
        }
