"""CPU tests for the node-local IPC transport slice-planning.

Only collect_overlapping_teacher_shards is CPU-testable — it's pure seq/vocab
overlap arithmetic that plans where each teacher shard lands in the student
rank's [T_t/CP_s, V_t] destination. The assemble / zero-copy / rebuild functions
read CUDA-IPC handles (GPU-only → HV-deferred).
"""

from dockyard_rl.algorithms.x_token.loss_utils import (
    collect_overlapping_teacher_shards,
)


def _shard(*, v0, v1, seq_start, seq_len, full_seq_len=4, full_vocab=6, vocab_local=None):
    return {
        "vocab_start_index": v0,
        "vocab_end_index": v1,
        "global_seq_start": seq_start,
        "actual_shape": (seq_len, vocab_local if vocab_local is not None else v1 - v0),
        "full_seq_len": full_seq_len,
        "full_vocab_size": full_vocab,
    }


def test_vocab_sharded_teacher_two_shards_cover_full_vocab():
    # Teacher TP=2: vocab split [0,3) and [3,6); full seq 4; student CP=1.
    shards = [
        _shard(v0=0, v1=3, seq_start=0, seq_len=4),
        _shard(v0=3, v1=6, seq_start=0, seq_len=4),
    ]
    m = collect_overlapping_teacher_shards(
        shards, student_cp_rank=0, student_cp_size=1, full_seq_len=4
    )
    assert len(m) == 2
    (_, ss0, sv0, ds0, dv0), (_, ss1, sv1, ds1, dv1) = m
    # full seq window on both; vocab dest slabs are the shard's global range.
    assert (ds0.start, ds0.stop) == (0, 4) and (ss0.start, ss0.stop) == (0, 4)
    assert (dv0.start, dv0.stop) == (0, 3)
    assert (dv1.start, dv1.stop) == (3, 6)
    # src_vocab is local (0..local_width).
    assert (sv0.start, sv0.stop) == (0, 3) and (sv1.start, sv1.stop) == (0, 3)


def test_cp_window_slices_src_and_dest():
    # Student CP=2, rank 1 owns seq window [2,4); a full-seq teacher shard maps
    # its src rows 2:4 into the student's dest rows 0:2.
    shards = [_shard(v0=0, v1=6, seq_start=0, seq_len=4)]
    m = collect_overlapping_teacher_shards(
        shards, student_cp_rank=1, student_cp_size=2, full_seq_len=4
    )
    assert len(m) == 1
    (_, ss, sv, ds, dv), = m
    assert (ss.start, ss.stop) == (2, 4)
    assert (ds.start, ds.stop) == (0, 2)
    assert (dv.start, dv.stop) == (0, 6)


def test_non_overlapping_seq_shard_skipped():
    # Shard's seq range [10,12) doesn't intersect the student window [0,4).
    shards = [_shard(v0=0, v1=6, seq_start=10, seq_len=2)]
    m = collect_overlapping_teacher_shards(
        shards, student_cp_rank=0, student_cp_size=1, full_seq_len=4
    )
    assert m == []


def test_partial_seq_overlap_clips_to_intersection():
    # Student CP=2 rank 0 owns [0,2); a teacher shard covering [1,4) overlaps on
    # [1,2): src rows 0:1 (shard-local), dest rows 1:2.
    shards = [_shard(v0=0, v1=6, seq_start=1, seq_len=3)]
    m = collect_overlapping_teacher_shards(
        shards, student_cp_rank=0, student_cp_size=2, full_seq_len=4
    )
    assert len(m) == 1
    (_, ss, sv, ds, dv), = m
    assert (ss.start, ss.stop) == (0, 1)  # overlap_start 1 - teacher_start 1
    assert (ds.start, ds.stop) == (1, 2)  # overlap_start 1 - student_start 0


def test_empty_shard_list_returns_empty_plan():
    assert collect_overlapping_teacher_shards(
        [], student_cp_rank=0, student_cp_size=1, full_seq_len=4
    ) == []
