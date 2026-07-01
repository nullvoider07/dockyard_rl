"""Collator that tokenizes the student once, then tokenizes+aligns each teacher.

The collator runs inside DataLoader worker processes. It does:

1. Tokenizes the source text once with the student tokenizer (no chat template,
   no special handling); the student tokenization is shared across all teachers.
2. For each *cross-tokenizer* teacher, tokenizes with that teacher's tokenizer
   and calls :class:`TokenAligner.align` to produce a dense-padded
   :class:`AlignmentBatch` (P-KL, gold_loss, xtoken_loss), emitting
   teacher-indexed keys ``teacher_{i}_*`` / ``alignment_{i}_*``.
3. *Same-tokenizer* teachers (``aligners[i] is None``) emit nothing extra — their
   forward reuses the student tokenization (identity 1:1 alignment).
4. Returns a :class:`BatchedDataDict` with the student keys the policy train step
   expects (``input_ids``, ``input_lengths``, ``token_mask``, ``sample_mask``)
   plus the per-teacher tensors and alignment tensors.

Loss-side projection-matrix work happens inside the loss fn; nothing related to
KL/CE math runs here.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any, List, Optional, cast

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dockyard_rl.algorithms.x_token.token_aligner import TokenAligner
from dockyard_rl.data.interfaces import DatumSpec
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict


class CrossTokenizerCollator:
    """Tokenize the student once, tokenize+align each teacher, return a flat batch.

    Supports N teachers. The student text is tokenized once and shared; each
    cross-tokenizer teacher is tokenized with its own tokenizer and aligned with
    its own :class:`TokenAligner`, emitting teacher-indexed keys (``teacher_{i}_*``
    and ``alignment_{i}_*``). A *same-tokenizer* teacher (``aligners[i] is None``)
    reuses the student tokenization and emits nothing extra.

    Args:
        student_tokenizer: HF tokenizer matching the student model.
        teacher_tokenizers: Per-teacher HF tokenizers. May be ``None`` for a
            same-tokenizer teacher (its tokenization is the student's).
        aligners: Per-teacher :class:`TokenAligner`. ``None`` marks a
            same-tokenizer teacher (no projection / no alignment).
        ctx_length_student: Hard tokenization length cap on the student side
            (also the padded sequence length of the student tensor).
        ctx_length_teachers: Per-teacher tokenization length caps.
        make_seq_div_by_student: Round student sequence length up to a multiple of
            this value (typically TP * CP * 2 for the DTensor V2 worker).
        make_seq_div_by_teachers: Per-teacher sequence-length divisors (defaults
            to 1 for each teacher when omitted).
    """

    def __init__(
        self,
        *,
        student_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizers: List[Optional[PreTrainedTokenizerBase]],
        aligners: List[Optional[TokenAligner]],
        ctx_length_student: int,
        ctx_length_teachers: List[int],
        make_seq_div_by_student: int = 1,
        make_seq_div_by_teachers: Optional[List[int]] = None,
    ):
        n = len(aligners)
        assert len(teacher_tokenizers) == n and len(ctx_length_teachers) == n, (
            "teacher_tokenizers, aligners, and ctx_length_teachers must all have "
            f"length == num_teachers ({n})."
        )
        if make_seq_div_by_teachers is None:
            make_seq_div_by_teachers = [1] * n
        assert len(make_seq_div_by_teachers) == n, (
            "make_seq_div_by_teachers must have length == num_teachers "
            f"({n}); got {len(make_seq_div_by_teachers)}."
        )
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizers = teacher_tokenizers
        self.aligners = aligners
        self.ctx_length_student = ctx_length_student
        self.ctx_length_teachers = ctx_length_teachers
        self.make_seq_div_by_student = make_seq_div_by_student
        self.make_seq_div_by_teachers = make_seq_div_by_teachers
        # Downstream consumers assume real tokens occupy the leading positions:
        # ``input_lengths = attention_mask.sum(-1)`` plus the ``[:length]`` slices
        # in the policy forward and the token-chunk alignment all treat
        # ``input_ids[:, :length]`` as the content. Pin right-padding rather than
        # trust each tokenizer's default (some configs default to left-padding,
        # which would silently misalign without changing the lengths).
        self._pin_padding(self.student_tokenizer)
        for tok in self.teacher_tokenizers:
            if tok is not None:
                self._pin_padding(tok)

    @staticmethod
    def _pin_padding(tokenizer: PreTrainedTokenizerBase) -> None:
        tokenizer.padding_side = "right"
        # Defensive: HF tokenizers without a pad token can't pad batches.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        # The data processor carries the raw text as a single assistant message;
        # the collator tokenizes that content for the student and each
        # cross-tokenizer teacher.
        texts = [cast(str, datum["message_log"][0]["content"]) for datum in batch]
        student_input_ids, student_attention_mask = self._tokenize_batch(
            texts,
            self.student_tokenizer,
            self.ctx_length_student,
            self.make_seq_div_by_student,
        )

        sample_mask = torch.tensor(
            [datum["loss_multiplier"] for datum in batch], dtype=torch.float32
        )
        idx = [datum["idx"] for datum in batch]

        out: dict[str, Any] = {
            # Student-side keys map onto the policy train step's expected names; a
            # single student tokenization is shared across all teachers.
            "input_ids": student_input_ids,
            "input_lengths": student_attention_mask.sum(dim=-1).long(),
            "token_mask": student_attention_mask.long(),
            "sample_mask": sample_mask,
            "idx": idx,
        }

        for i, aligner in enumerate(self.aligners):
            if aligner is None:
                # Same-tokenizer teacher: no re-tokenization, no projection, no
                # alignment. Its forward reuses the student tokenization.
                continue
            teacher_tokenizer = self.teacher_tokenizers[i]
            assert teacher_tokenizer is not None, (
                f"teacher {i} has an aligner but no tokenizer; a cross-tokenizer "
                "teacher needs both."
            )
            teacher_input_ids, teacher_attention_mask = self._tokenize_batch(
                texts,
                teacher_tokenizer,
                self.ctx_length_teachers[i],
                self.make_seq_div_by_teachers[i],
            )
            alignment = aligner.align(
                student_input_ids,
                teacher_input_ids,
                student_attention_mask=student_attention_mask,
                teacher_attention_mask=teacher_attention_mask,
            )
            # Teacher-side keys travel with the batch for the teacher forward.
            out[f"teacher_{i}_input_ids"] = teacher_input_ids
            out[f"teacher_{i}_input_lengths"] = teacher_attention_mask.sum(
                dim=-1
            ).long()
            out[f"teacher_{i}_token_mask"] = teacher_attention_mask.long()
            # Alignment payload, dense-padded so the DTensor policy can shard on
            # dim 0. Keys are driven off the AlignmentBatch fields so they can't
            # drift from ``alignment_from_flat_batch(data, prefix=f"alignment_{i}_")``.
            for f in dataclass_fields(alignment):
                out[f"alignment_{i}_{f.name}"] = getattr(alignment, f.name)

        return BatchedDataDict(out)

    @staticmethod
    def _tokenize_batch(
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        ctx_length: int,
        make_seq_div_by: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch and pad to a multiple of ``make_seq_div_by``."""
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=ctx_length,
            return_tensors="pt",
        )
        input_ids: torch.Tensor = encoded["input_ids"]
        attention_mask: torch.Tensor = encoded["attention_mask"]

        b, t = input_ids.shape
        pad = (make_seq_div_by - (t % make_seq_div_by)) % make_seq_div_by
        if pad > 0:
            pad_ids = torch.full(
                (b, pad),
                cast(int, tokenizer.pad_token_id),
                dtype=input_ids.dtype,
            )
            pad_mask = torch.zeros((b, pad), dtype=attention_mask.dtype)
            input_ids = torch.cat([input_ids, pad_ids], dim=1)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

        return input_ids, attention_mask
