"""CPU tests for the cross-cluster teacher-logit transport kernel.

The encode (bf16 + seq-chunk) / decode (concat + CP-slice + fp32) round-trip is
pure tensor logic and CPU round-trip testable; the data_plane put/get wiring is
GPU/cluster integration (HV-deferred). The consumer reassembles to the same
[B, T_t/CP_s, V_t] contract the IPC path produces, so the loss stays
transport-blind.
"""

import torch

from dockyard_rl.algorithms.x_token.loss_utils import (
    chunk_teacher_logits_for_cross_cluster,
    rebuild_teacher_full_logits_cross_cluster,
)


def test_round_trip_world1_matches_bf16_cast():
    tl = torch.randn(2, 8, 6)
    chunks = chunk_teacher_logits_for_cross_cluster(tl, num_seq_chunks=4)
    rt = rebuild_teacher_full_logits_cross_cluster(chunks, cp_group=None)
    assert rt.shape == (2, 8, 6) and rt.dtype == torch.float32
    # The only lossy step is the bf16 cast; decode reproduces it exactly.
    assert torch.equal(rt, tl.to(torch.bfloat16).float())


def test_chunks_are_bf16_and_split_along_seq():
    chunks = chunk_teacher_logits_for_cross_cluster(torch.randn(2, 8, 6), num_seq_chunks=4)
    assert len(chunks) == 4
    assert all(c.dtype == torch.bfloat16 for c in chunks)
    assert all(c.shape == (2, 2, 6) for c in chunks)
    assert all(c.is_contiguous() for c in chunks)


def test_uneven_seq_split_preserves_total_length():
    chunks = chunk_teacher_logits_for_cross_cluster(torch.randn(1, 7, 4), num_seq_chunks=3)
    # torch.chunk(7, 3) -> [3, 3, 1]
    assert [c.shape[1] for c in chunks] == [3, 3, 1]
    rt = rebuild_teacher_full_logits_cross_cluster(chunks, cp_group=None)
    assert rt.shape == (1, 7, 4)


def test_single_chunk_is_identity_round_trip():
    tl = torch.randn(1, 5, 3)
    chunks = chunk_teacher_logits_for_cross_cluster(tl, num_seq_chunks=1)
    assert len(chunks) == 1
    rt = rebuild_teacher_full_logits_cross_cluster(chunks, cp_group=None)
    assert torch.equal(rt, tl.to(torch.bfloat16).float())


def test_chunk_count_below_one_raises():
    try:
        chunk_teacher_logits_for_cross_cluster(torch.randn(1, 4, 2), num_seq_chunks=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_rebuild_empty_chunks_raises():
    try:
        rebuild_teacher_full_logits_cross_cluster([], cp_group=None)
        assert False, "expected ValueError"
    except ValueError:
        pass
