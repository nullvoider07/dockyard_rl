"""CPU tests for the multi-teacher cross-tokenizer loss-input keystone.

prepare_xtoken_cross_tokenizer_loss_input rebuilds each teacher's logits from the
transport and does the CP-resolution glue. The teacher rebuild + cuda device are
GPU-only, so they're monkeypatched; the world=1 / non-DTensor glue (group
derivation, identity CP relayout, next-token chunk-id shift, and the shared
student_input_ids / token_mask population the accuracy metric requires) is
exercised on CPU. Covers a cross-tokenizer teacher, a same-tokenizer teacher
(thin identity alignment), and two teachers.
"""

from typing import Any

import torch

import dockyard_rl.algorithms.x_token.loss_utils as lu
from dockyard_rl.algorithms.x_token.loss_utils import (
    LocalizedAlignment,
    prepare_xtoken_cross_tokenizer_loss_input,
)


def _patch_transport(monkeypatch, teacher_logits_by_idx):
    """Patch the GPU-only IPC rebuild to hand back per-teacher tensors by key."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def _fake_rebuild(entries, cp_group, device):
        # `entries` is the list stored under teacher_{i}_full_logits_ipc; the test
        # tags each with its teacher index via the sentinel below.
        return teacher_logits_by_idx[entries[0]["t"]]

    monkeypatch.setattr(lu, "rebuild_teacher_full_logits_from_ipc", _fake_rebuild)


def _teacher_ipc(i, b=1):
    return [{"t": i} for _ in range(b)]


def _cross_tok_data(i, b=1, ts=4) -> dict[str, Any]:
    """The teacher_{i}_* + alignment_{i}_* keys a cross-tokenizer teacher needs."""
    return {
        f"teacher_{i}_full_logits_ipc": _teacher_ipc(i, b),
        f"alignment_{i}_student_chunk_id": torch.tensor([[0, 1, 2, -1]] * b),
        f"alignment_{i}_teacher_chunk_id": torch.tensor([[0, 1, 2, -1]] * b),
        f"alignment_{i}_pair_valid": torch.ones(b, 3, dtype=torch.bool),
        f"alignment_{i}_pair_is_correct": torch.ones(b, 3, dtype=torch.bool),
    }


def _student_data(b=1, ts=4) -> dict[str, Any]:
    return {
        "input_ids": torch.randint(0, 9, (b, ts)),
        "token_mask": torch.ones(b, ts),
        "sample_mask": torch.ones(b),
    }


def _one_cross_tok(b=1, ts=4) -> dict[str, Any]:
    data = _student_data(b, ts)
    data.update(_cross_tok_data(0, b, ts))
    return data


def test_keystone_non_dtensor_world1_glue(monkeypatch):
    teacher = torch.randn(1, 4, 6)
    _patch_transport(monkeypatch, {0: teacher})
    logits = torch.randn(1, 4, 8)
    student_contig, tfl_by_idx, aligns_by_idx, tp_group, cp_group = (
        prepare_xtoken_cross_tokenizer_loss_input(
            logits, _one_cross_tok(), projection_matrix_paths=["/p0"],
            vocab_parallel_group=None, context_parallel_group=None,
        )
    )
    assert tfl_by_idx[0] is teacher
    # world=1 cp relayout is identity.
    assert torch.equal(student_contig, logits)
    assert isinstance(aligns_by_idx[0], LocalizedAlignment)
    assert tp_group is None and cp_group is None


def test_keystone_sets_shared_student_fields(monkeypatch):
    teacher = torch.randn(1, 4, 6)
    _patch_transport(monkeypatch, {0: teacher})
    logits = torch.randn(1, 4, 8)
    data = _one_cross_tok()
    _, _, aligns_by_idx, _, _ = prepare_xtoken_cross_tokenizer_loss_input(
        logits, data, projection_matrix_paths=["/p0"],
        vocab_parallel_group=None, context_parallel_group=None,
    )
    align = aligns_by_idx[0]
    # student_input_ids / token_mask populated (the accuracy metric + same-vocab KD
    # read them; localize_alignment leaves them None).
    assert align.student_input_ids is not None
    assert align.student_token_mask is not None
    assert torch.equal(align.student_input_ids, data["input_ids"])  # world=1 identity
    assert torch.equal(align.student_token_mask, data["token_mask"])


def test_keystone_next_token_shifts_chunk_ids(monkeypatch):
    teacher = torch.randn(1, 4, 6)
    _patch_transport(monkeypatch, {0: teacher})
    logits = torch.randn(1, 4, 8)
    _, _, aligns_by_idx, _, _ = prepare_xtoken_cross_tokenizer_loss_input(
        logits, _one_cross_tok(), projection_matrix_paths=["/p0"],
        vocab_parallel_group=None, context_parallel_group=None,
    )
    align = aligns_by_idx[0]
    assert align.student_chunk_id is not None and align.teacher_chunk_id is not None
    # [0,1,2,-1] -> next-token (left-1) shift, fill -1 at the boundary.
    assert align.student_chunk_id.tolist() == [[1, 2, -1, -1]]
    assert align.teacher_chunk_id.tolist() == [[1, 2, -1, -1]]


def test_keystone_same_tokenizer_thin_alignment(monkeypatch):
    # projection None -> same-tokenizer teacher: full logits rebuilt, but a thin
    # identity alignment (no chunk fields) carrying only the shared student data.
    teacher = torch.randn(1, 4, 8)
    _patch_transport(monkeypatch, {0: teacher})
    logits = torch.randn(1, 4, 8)
    data = _student_data()
    data["teacher_0_full_logits_ipc"] = _teacher_ipc(0)  # no alignment_0_* keys
    _, tfl_by_idx, aligns_by_idx, _, _ = prepare_xtoken_cross_tokenizer_loss_input(
        logits, data, projection_matrix_paths=[None],
        vocab_parallel_group=None, context_parallel_group=None,
    )
    align = aligns_by_idx[0]
    assert tfl_by_idx[0] is teacher
    assert align.student_input_ids is not None and align.student_token_mask is not None
    # thin alignment: no chunk / pair fields.
    assert align.student_chunk_id is None and align.pair_valid is None


def test_keystone_two_teachers(monkeypatch):
    t0, t1 = torch.randn(1, 4, 6), torch.randn(1, 4, 8)
    _patch_transport(monkeypatch, {0: t0, 1: t1})
    logits = torch.randn(1, 4, 8)
    data = _student_data()
    data.update(_cross_tok_data(0))       # teacher 0 = cross-tokenizer
    data["teacher_1_full_logits_ipc"] = _teacher_ipc(1)  # teacher 1 = same-tokenizer
    _, tfl_by_idx, aligns_by_idx, _, _ = prepare_xtoken_cross_tokenizer_loss_input(
        logits, data, projection_matrix_paths=["/p0", None],
        vocab_parallel_group=None, context_parallel_group=None,
    )
    assert set(tfl_by_idx) == {0, 1} and set(aligns_by_idx) == {0, 1}
    assert aligns_by_idx[0].student_chunk_id is not None       # cross-tok: chunks set
    assert aligns_by_idx[1].student_chunk_id is None           # same-tok: thin


def test_keystone_cross_cluster_branch_end_to_end():
    # The cross-cluster branch reassembles bf16 seq-chunks (no IPC / cuda), so the
    # full keystone runs on CPU end-to-end when data carries the cross-cluster key.
    from dockyard_rl.algorithms.x_token.loss_utils import (
        chunk_teacher_logits_for_cross_cluster,
    )

    teacher = torch.randn(1, 4, 6)
    data = _student_data()
    data.update(_cross_tok_data(0))
    del data["teacher_0_full_logits_ipc"]
    data["teacher_0_full_logits_cross_cluster"] = (
        chunk_teacher_logits_for_cross_cluster(teacher, num_seq_chunks=2)
    )
    logits = torch.randn(1, 4, 8)
    student_contig, tfl_by_idx, aligns_by_idx, _, _ = (
        prepare_xtoken_cross_tokenizer_loss_input(
            logits, data, projection_matrix_paths=["/p0"],
            vocab_parallel_group=None, context_parallel_group=None,
        )
    )
    assert torch.equal(tfl_by_idx[0], teacher.to(torch.bfloat16).float())
    assert torch.equal(student_contig, logits)
    assert aligns_by_idx[0].student_input_ids is not None
    assert aligns_by_idx[0].student_chunk_id is not None
    assert aligns_by_idx[0].student_chunk_id.tolist() == [[1, 2, -1, -1]]


def test_keystone_missing_teacher_logits_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    data = _student_data()  # neither transport key for teacher 0
    try:
        prepare_xtoken_cross_tokenizer_loss_input(
            torch.randn(1, 4, 8), data, projection_matrix_paths=["/p0"],
            vocab_parallel_group=None, context_parallel_group=None,
        )
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_keystone_falls_back_to_passed_tp_group_for_plain_logits(monkeypatch):
    # Non-DTensor logits => tp_group comes from the passed vocab_parallel_group
    # (only returned, never used in a distributed op here). cp_group stays None on
    # CPU since the body uses it in world-size checks.
    teacher = torch.randn(1, 4, 6)
    _patch_transport(monkeypatch, {0: teacher})
    sentinel_tp = object()
    _, _, _, tp_group, cp_group = prepare_xtoken_cross_tokenizer_loss_input(
        torch.randn(1, 4, 8), _one_cross_tok(), projection_matrix_paths=["/p0"],
        vocab_parallel_group=sentinel_tp, context_parallel_group=None,
    )
    assert tp_group is sentinel_tp and cp_group is None
