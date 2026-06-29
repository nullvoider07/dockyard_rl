"""CPU tests for the xtoken CP-localized CE / accuracy helpers (M2.c).

Covers the single-rank / non-DTensor paths of student_next_token_ce,
ce_label_mask, next_token_accuracy, and localize_alignment. The DTensor and
multi-rank CP branches are GPU-deferred.
"""

import torch

from dockyard_rl.algorithms.x_token.loss_utils import (
    LocalizedAlignment,
    ce_label_mask,
    localize_alignment,
    next_token_accuracy,
    student_next_token_ce,
)


# -- student_next_token_ce ----------------------------------------------------

def test_student_next_token_ce_matches_manual_shifted_ce():
    b, t, v = 2, 5, 7
    logits = torch.randn(b, t, v)
    input_ids = torch.randint(0, v, (b, t))
    ce = student_next_token_ce(logits, input_ids=input_ids)
    assert ce.shape == (b, t - 1)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, v).float(),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(b, t - 1)
    assert torch.allclose(ce, expected, atol=1e-5)


# -- ce_label_mask ------------------------------------------------------------

def test_ce_label_mask_shifts_and_gates_by_sample_mask():
    b, t = 2, 5
    token_mask = torch.ones(b, t)
    token_mask[0, 0] = 0  # first token masked (dropped by the 1: shift anyway)
    sample_mask = torch.tensor([1.0, 0.0])
    m = ce_label_mask(
        token_mask=token_mask, sample_mask=sample_mask, ce_seq_len=t - 1,
        dtype=torch.float32,
    )
    assert m.shape == (b, t - 1)
    # sample 1 fully masked out; sample 0 = token_mask[:, 1:t] all ones.
    assert m[1].sum() == 0
    assert torch.equal(m[0], torch.ones(t - 1))


# -- next_token_accuracy ------------------------------------------------------

def test_next_token_accuracy_perfect_predictor_is_one():
    # input_ids -> next labels (shift, fill 0) = [1, 2, 3, 0]; the last position
    # is masked out (next_mask shift fill 0), so a predictor whose argmax equals
    # the next token at positions 0..2 scores 1.0.
    input_ids = torch.tensor([[5, 1, 2, 3]])
    next_labels = [1, 2, 3]
    logits = torch.zeros(1, 4, 6)
    for i, lab in enumerate(next_labels):
        logits[0, i, lab] = 10.0
    acc = next_token_accuracy(
        logits, input_ids=input_ids,
        token_mask=torch.ones(1, 4), sample_mask=torch.tensor([1.0]),
    )
    assert abs(float(acc) - 1.0) < 1e-6


def test_next_token_accuracy_half_correct():
    input_ids = torch.tensor([[5, 1, 2, 3]])  # next labels at pos 0..2: 1,2,3
    logits = torch.zeros(1, 4, 6)
    logits[0, 0, 1] = 10.0  # correct
    logits[0, 1, 0] = 10.0  # wrong (label 2)
    logits[0, 2, 3] = 10.0  # correct
    acc = next_token_accuracy(
        logits, input_ids=input_ids,
        token_mask=torch.ones(1, 4), sample_mask=torch.tensor([1.0]),
    )
    assert abs(float(acc) - (2.0 / 3.0)) < 1e-6


def test_next_token_accuracy_sample_mask_zero_gives_zero_denom_clamp():
    input_ids = torch.tensor([[5, 1, 2, 3]])
    logits = torch.zeros(1, 4, 6)
    logits[0, 0, 1] = 10.0
    acc = next_token_accuracy(
        logits, input_ids=input_ids,
        token_mask=torch.ones(1, 4), sample_mask=torch.tensor([0.0]),
    )
    # all masked => correct 0 / denom clamp(min=1) => 0.0, no NaN.
    assert float(acc) == 0.0


# -- localize_alignment -------------------------------------------------------

def test_localize_alignment_unwraps_and_slices_teacher_window():
    data = {
        "alignment_teacher_chunk_id": torch.arange(6).reshape(1, 6),
        "alignment_student_chunk_id": torch.arange(4).reshape(1, 4),
        "alignment_pair_valid": torch.ones(1, 3, dtype=torch.bool),
        "alignment_pair_is_correct": torch.zeros(1, 3, dtype=torch.bool),
        "sample_mask": torch.tensor([1.0]),
    }
    la = localize_alignment(data, teacher_seq_len=6, cp_group=None)
    assert isinstance(la, LocalizedAlignment)
    # world=1 => cp_rank 0 => full teacher window.
    assert torch.equal(la.teacher_chunk_id, torch.arange(6).reshape(1, 6))
    assert torch.equal(la.student_chunk_id, torch.arange(4).reshape(1, 4))
    assert la.student_input_ids is None and la.student_token_mask is None
