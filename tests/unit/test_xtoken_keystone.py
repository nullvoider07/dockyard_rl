"""CPU tests for the cross-tokenizer loss-input keystone (M3.c).

prepare_xtoken_cross_tokenizer_loss_input rebuilds teacher logits from the
transport and does the CP-resolution glue. The teacher rebuild + cuda device are
GPU-only, so they're monkeypatched; the world=1 / non-DTensor glue (group
derivation, identity CP relayout, next-token chunk-id shift, Contract 3
student_input_ids / token_mask population) is exercised on CPU.
"""

import torch

import dockyard_rl.algorithms.x_token.loss_utils as lu
from dockyard_rl.algorithms.x_token.loss_utils import (
    LocalizedAlignment,
    prepare_xtoken_cross_tokenizer_loss_input,
)


def _patch_transport(monkeypatch, teacher_logits):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        lu, "rebuild_teacher_full_logits_from_ipc",
        lambda entries, cp_group, device: teacher_logits,
    )


def _data(b=1, ts=4):
    return {
        "teacher_full_logits_ipc": [{} for _ in range(b)],  # ignored (patched)
        "input_ids": torch.randint(0, 9, (b, ts)),
        "token_mask": torch.ones(b, ts),
        "sample_mask": torch.ones(b),
        "alignment_student_chunk_id": torch.tensor([[0, 1, 2, -1]] * b),
        "alignment_teacher_chunk_id": torch.tensor([[0, 1, 2, -1]] * b),
        "alignment_pair_valid": torch.ones(b, 3, dtype=torch.bool),
        "alignment_pair_is_correct": torch.ones(b, 3, dtype=torch.bool),
    }


def test_keystone_non_dtensor_world1_glue(monkeypatch):
    teacher = torch.randn(1, 4, 6)
    _patch_transport(monkeypatch, teacher)
    logits = torch.randn(1, 4, 8)
    data = _data()
    tfl, student_contig, align, tp_group, cp_group = (
        prepare_xtoken_cross_tokenizer_loss_input(
            logits, data, vocab_parallel_group=None, context_parallel_group=None,
        )
    )
    assert tfl is teacher
    # world=1 cp relayout is identity.
    assert torch.equal(student_contig, logits)
    assert isinstance(align, LocalizedAlignment)
    assert tp_group is None and cp_group is None


def test_keystone_sets_contract3_student_fields(monkeypatch):
    teacher = torch.randn(1, 4, 6)
    _patch_transport(monkeypatch, teacher)
    logits = torch.randn(1, 4, 8)
    data = _data()
    _, _, align, _, _ = prepare_xtoken_cross_tokenizer_loss_input(
        logits, data, vocab_parallel_group=None, context_parallel_group=None,
    )
    # Contract 3: student_input_ids / token_mask populated (the P-KL accuracy
    # metric reads them; localize_alignment leaves them None).
    assert align.student_input_ids is not None
    assert align.student_token_mask is not None
    assert torch.equal(align.student_input_ids, data["input_ids"])  # world=1 identity
    assert torch.equal(align.student_token_mask, data["token_mask"])


def test_keystone_next_token_shifts_chunk_ids(monkeypatch):
    teacher = torch.randn(1, 4, 6)
    _patch_transport(monkeypatch, teacher)
    logits = torch.randn(1, 4, 8)
    data = _data()
    _, _, align, _, _ = prepare_xtoken_cross_tokenizer_loss_input(
        logits, data, vocab_parallel_group=None, context_parallel_group=None,
    )
    # [0,1,2,-1] -> next-token (left-1) shift, fill -1 at the boundary.
    assert align.student_chunk_id.tolist() == [[1, 2, -1, -1]]
    assert align.teacher_chunk_id.tolist() == [[1, 2, -1, -1]]


def test_keystone_falls_back_to_passed_tp_group_for_plain_logits(monkeypatch):
    # Non-DTensor logits => tp_group comes from the passed vocab_parallel_group
    # (tp_group is only returned, never used in a distributed op here, so a
    # sentinel is safe). cp_group must stay None on CPU since the body uses it
    # in world-size checks.
    teacher = torch.randn(1, 4, 6)
    _patch_transport(monkeypatch, teacher)
    sentinel_tp = object()
    _, _, _, tp_group, cp_group = prepare_xtoken_cross_tokenizer_loss_input(
        torch.randn(1, 4, 8), _data(),
        vocab_parallel_group=sentinel_tp, context_parallel_group=None,
    )
    assert tp_group is sentinel_tp and cp_group is None
