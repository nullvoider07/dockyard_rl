"""CPU tests for the xtoken loss math.

These cover the single-rank, transport-free pieces of the cross-tokenizer
distillation loss: the FP32 sparse-dense matmul autograd function, the
chunk-average reductions, the sparse-projection contraction, top-k teacher
selection, the projection-file loaders/caches, the common/uncommon exact-token
partition, and the flat-batch rehydrator. The TP/CP-collective and CUDA-IPC
paths are GPU-only and validate on hardware.
"""

import torch

from dockyard_rl.algorithms.x_token import loss_utils as lu
from dockyard_rl.algorithms.x_token.token_aligner import AlignmentBatch


def _coo(rows, cols, vals, shape):
    return torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long),
        torch.tensor(vals, dtype=torch.float32),
        shape,
    ).coalesce()


# -- Fp32SparseMM -------------------------------------------------------------

def test_fp32_sparse_mm_forward_matches_dense():
    M = _coo([0, 1, 2], [0, 0, 1], [1.0, 2.0, 3.0], (3, 2))  # [V_s, V_t]
    dense = torch.randn(3, 4)
    out = lu.Fp32SparseMM.apply(M, dense)  # M.t() @ dense -> [2, 4]
    expected = M.to_dense().t() @ dense
    assert out.shape == (2, 4)
    assert torch.allclose(out, expected, atol=1e-6)


def test_fp32_sparse_mm_backward_grad_dense_and_none_sparse():
    M = _coo([0, 1, 2], [0, 1, 1], [1.0, 2.0, 3.0], (3, 2))
    dense = torch.randn(3, 4, requires_grad=True)
    out = lu.Fp32SparseMM.apply(M, dense)
    grad_out = torch.randn(2, 4)
    out.backward(grad_out)
    # d/d_dense = M @ grad_out
    expected_grad = (M.to_dense() @ grad_out)
    assert dense.grad is not None
    assert torch.allclose(dense.grad, expected_grad, atol=1e-6)


def test_fp32_sparse_mm_upcasts_fp32_under_no_autocast():
    # On CPU the custom_fwd cast still forces fp32 output dtype.
    M = _coo([0, 1], [0, 1], [1.0, 1.0], (2, 2))
    dense = torch.randn(2, 3, dtype=torch.float32)
    out = lu.Fp32SparseMM.apply(M, dense)
    assert out.dtype == torch.float32


# -- chunk aggregation --------------------------------------------------------

def test_chunk_log_prob_sums_buckets_and_excludes_negative_one():
    lp = torch.arange(2 * 4 * 1, dtype=torch.float32).reshape(2, 4, 1)
    # sample 0: tokens [0,1,2,3] -> chunks [0,0,1,-1]
    # sample 1: tokens [4,5,6,7] -> chunks [-1,1,1,1]
    cid = torch.tensor([[0, 0, 1, -1], [-1, 1, 1, 1]])
    sums, sizes = lu.chunk_log_prob_sums(lp, cid, max_chunks=2)
    assert sums.shape == (2, 2, 1)
    assert sizes.shape == (2, 2)
    # sample 0 chunk 0: tokens 0+1 = 1; chunk 1: token 2 = 2
    assert sums[0, 0, 0].item() == 1.0 and sums[0, 1, 0].item() == 2.0
    assert sizes[0, 0].item() == 2.0 and sizes[0, 1].item() == 1.0
    # sample 1 chunk 0: empty; chunk 1: 5+6+7 = 18
    assert sizes[1, 0].item() == 0.0 and sizes[1, 1].item() == 3.0
    assert sums[1, 1, 0].item() == 18.0


def test_chunk_average_finalize_divides_and_guards_empty():
    sums = torch.tensor([[[4.0], [0.0]]])  # [1, 2, 1]
    sizes = torch.tensor([[2.0, 0.0]])
    avg, out_sizes = lu.chunk_average_finalize(sums, sizes)
    assert torch.allclose(avg[0, 0], torch.tensor([2.0]))
    # empty bucket: 0 / (0 + eps) == 0, no NaN/Inf
    assert torch.isfinite(avg).all()
    assert avg[0, 1, 0].item() == 0.0
    assert torch.equal(out_sizes, sizes)


def test_chunk_average_log_probs_end_to_end_single_rank():
    lp = torch.randn(1, 3, 2)
    cid = torch.tensor([[0, 0, 1]])
    avg, _sizes = lu.chunk_average_log_probs(lp, cid, max_chunks=2)
    assert avg.shape == (1, 2, 2)
    assert torch.allclose(avg[0, 0], (lp[0, 0] + lp[0, 1]) / 2)
    assert torch.allclose(avg[0, 1], lp[0, 2])


# -- sparse projection slicing + contraction ----------------------------------

def test_slice_sparse_projection_rows_filters_and_shifts():
    M = _coo([0, 1, 2, 3], [0, 1, 0, 1], [1.0, 2.0, 3.0, 4.0], (4, 2))
    local = lu.slice_sparse_projection_rows(M, row_start=1, row_end=3)
    dense = local.to_dense()
    assert dense.shape == (2, 2)
    # original rows 1,2 -> local rows 0,1
    assert dense[0, 1].item() == 2.0  # was row 1 col 1
    assert dense[1, 0].item() == 3.0  # was row 2 col 0


def test_project_student_to_teacher_vocab_single_rank_matches_dense():
    sp = _coo([0, 1, 2], [0, 1, 0], [1.0, 1.0, 2.0], (3, 2))  # [V_s, V_t]
    probs = torch.rand(2, 5, 3)
    out = lu.project_student_to_teacher_vocab(probs, sp)
    assert out.shape == (2, 5, 2)
    expected = probs.reshape(-1, 3) @ sp.to_dense()
    assert torch.allclose(out.reshape(-1, 2), expected, atol=1e-5)


# -- top-k teacher selection --------------------------------------------------

def test_select_teacher_topk_indices_sorted_global_by_max():
    # vocab 6; craft per-vocab max importance ordering.
    tl = torch.full((1, 1, 6), -10.0)
    tl[0, 0, 1] = 5.0
    tl[0, 0, 3] = 4.0
    tl[0, 0, 5] = 3.0
    tl[0, 0, 0] = 2.0
    idx = lu.select_teacher_topk_indices(tl, k=3)
    assert idx.shape == (3,)
    # top-3 by importance = {1,3,5}, returned sorted ascending
    assert idx.tolist() == [1, 3, 5]


def test_select_teacher_topk_takes_max_over_flattened_bt():
    tl = torch.zeros(2, 2, 4)
    tl[1, 0, 2] = 9.0  # one position spikes vocab 2
    tl[0, 1, 0] = 1.0
    idx = lu.select_teacher_topk_indices(tl, k=1)
    assert idx.tolist() == [2]


# -- valid_chunk_mask ---------------------------------------------------------

def test_valid_chunk_mask_requires_both_sides_and_pair_valid():
    s = torch.tensor([1.0, 0.0, 2.0, 3.0])
    t = torch.tensor([1.0, 2.0, 0.0, 4.0])
    pv = torch.tensor([True, True, True, False])
    mask = lu.valid_chunk_mask(s, t, pv)
    assert mask.tolist() == [True, False, False, False]


# -- projection file parsing --------------------------------------------------

def _save_dense_projection(path, indices, likelihoods):
    torch.save(
        {"indices": torch.tensor(indices), "likelihoods": torch.tensor(likelihoods)},
        path,
    )


def test_parse_projection_file_dense_format(tmp_path):
    p = tmp_path / "dense.pt"
    # V_s=2, top_k=2; row0 -> teacher [3, -1], row1 -> teacher [1, 2]
    _save_dense_projection(p, [[3, -1], [1, 2]], [[1.0, 0.0], [0.6, 0.4]])
    indices, _values, v_s, v_t = lu.parse_projection_file(p)
    assert indices.shape == (2, 4)
    assert v_s == 2
    assert v_t == 4  # max positive teacher id (3) + 1
    # -1 sentinel preserved in indices
    assert (indices[1] == -1).any()


def test_parse_projection_file_sparse_tuple_format(tmp_path):
    p = tmp_path / "sparse.pt"
    torch.save({(0, 5): 3, (2, 1): 7}, p)
    _indices, values, v_s, v_t = lu.parse_projection_file(p)
    assert v_s == 3  # max student id 2 + 1
    assert v_t == 6  # max teacher id 5 + 1
    assert values.dtype == torch.float32


def test_parse_projection_file_missing_raises(tmp_path):
    try:
        lu.parse_projection_file(tmp_path / "nope.pt")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_parse_projection_file_unrecognized_raises(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"foo": torch.tensor([1])}, p)
    try:
        lu.parse_projection_file(p)
        assert False, "expected ValueError"
    except ValueError:
        pass


# -- sparse projection matrix cache -------------------------------------------

def test_get_sparse_projection_matrix_drops_sentinel_and_sizes(tmp_path):
    p = tmp_path / "dense2.pt"
    # row0 -> [3, -1]: the -1 must be dropped from the sparse matrix.
    _save_dense_projection(p, [[3, -1], [1, 2]], [[1.0, 0.0], [0.6, 0.4]])
    sp = lu.get_sparse_projection_matrix(
        p, torch.device("cpu"), student_vocab_size=10, teacher_vocab_size=8
    )
    assert sp.shape == (10, 8)  # sized up to configured vocabs
    # No negative column survives.
    assert (sp.indices()[1] >= 0).all()
    dense = sp.to_dense()
    assert dense[0, 3].item() == 1.0
    assert abs(dense[1, 1].item() - 0.6) < 1e-6 and abs(dense[1, 2].item() - 0.4) < 1e-6


def test_get_sparse_projection_matrix_caches(tmp_path):
    p = tmp_path / "dense3.pt"
    _save_dense_projection(p, [[0], [1]], [[1.0], [1.0]])
    a = lu.get_sparse_projection_matrix(
        p, torch.device("cpu"), student_vocab_size=2, teacher_vocab_size=2
    )
    b = lu.get_sparse_projection_matrix(
        p, torch.device("cpu"), student_vocab_size=2, teacher_vocab_size=2
    )
    assert a is b  # same object from cache


def test_get_topk_projection_rejects_sparse_format(tmp_path):
    p = tmp_path / "sparse2.pt"
    torch.save({(0, 1): 2}, p)
    try:
        lu.get_topk_projection(p, torch.device("cpu"))
        assert False, "expected ValueError"
    except ValueError:
        pass


# -- exact-token map partition ------------------------------------------------

def test_build_exact_token_map_strict(tmp_path):
    p = tmp_path / "strict.pt"
    # row0: weight 1.0 + second slot -1 => exact map to teacher 4.
    # row1: weight 0.7 (not 1.0) => not exact.
    # row2: weight 1.0 but has a second mapping (col1 != -1) => not exact.
    _save_dense_projection(
        p,
        [[4, -1], [2, 3], [1, 5]],
        [[1.0, 0.0], [0.7, 0.3], [1.0, 0.2]],
    )
    m = lu.build_exact_token_map(
        p, torch.device("cpu"), xtoken_loss=False, teacher_vocab_size=8
    )
    assert m["common_student"].tolist() == [0]
    assert m["common_teacher"].tolist() == [4]
    assert 0 not in m["uncommon_student"].tolist()
    assert 4 not in m["uncommon_teacher"].tolist()
    # uncommon covers the remaining vocab axes
    assert set(m["uncommon_student"].tolist()) == {1, 2}


def test_build_exact_token_map_relaxed_threshold(tmp_path):
    p = tmp_path / "relaxed.pt"
    # relaxed: first-weight >= 0.6 qualifies.
    _save_dense_projection(
        p,
        [[1, -1], [2, 3], [0, 5]],
        [[0.9, 0.0], [0.5, 0.3], [0.6, 0.1]],
    )
    m = lu.build_exact_token_map(
        p, torch.device("cpu"), xtoken_loss=True, teacher_vocab_size=8
    )
    # rows 0 (0.9) and 2 (0.6) qualify; row 1 (0.5) does not.
    assert m["common_student"].tolist() == [0, 2]
    assert m["common_teacher"].tolist() == [1, 0]


def test_build_exact_token_map_collision_lowest_student_wins_strict(tmp_path):
    p = tmp_path / "collide.pt"
    # both row0 and row1 map exactly to teacher 2 -> lowest student (0) wins.
    _save_dense_projection(
        p,
        [[2, -1], [2, -1]],
        [[1.0, 0.0], [1.0, 0.0]],
    )
    m = lu.build_exact_token_map(
        p, torch.device("cpu"), xtoken_loss=False, teacher_vocab_size=4
    )
    assert m["common_student"].tolist() == [0]
    assert m["common_teacher"].tolist() == [2]
    assert 1 in m["uncommon_student"].tolist()


def test_build_exact_token_map_empty_when_no_exact(tmp_path):
    p = tmp_path / "none.pt"
    _save_dense_projection(p, [[1, 2], [3, 4]], [[0.4, 0.3], [0.5, 0.2]])
    m = lu.build_exact_token_map(
        p, torch.device("cpu"), xtoken_loss=False, teacher_vocab_size=6
    )
    assert m["common_student"].numel() == 0
    assert m["common_teacher"].numel() == 0
    assert m["uncommon_student"].tolist() == [0, 1]
    assert m["uncommon_teacher"].tolist() == list(range(6))


def test_build_exact_token_map_caches(tmp_path):
    p = tmp_path / "cache.pt"
    _save_dense_projection(p, [[1, -1]], [[1.0, 0.0]])
    a = lu.build_exact_token_map(
        p, torch.device("cpu"), xtoken_loss=False, teacher_vocab_size=4
    )
    b = lu.build_exact_token_map(
        p, torch.device("cpu"), xtoken_loss=False, teacher_vocab_size=4
    )
    assert a is b


# -- flat-batch rehydration ---------------------------------------------------

def test_alignment_from_flat_batch_rebuilds_schema():
    B, T_s, T_t, P = 2, 5, 4, 3
    data = {
        "alignment_pair_valid": torch.ones(B, P, dtype=torch.bool),
        "alignment_pair_is_correct": torch.zeros(B, P, dtype=torch.bool),
        "alignment_student_exact_partition_mask": torch.zeros(B, T_s, dtype=torch.bool),
        "alignment_teacher_exact_partition_mask": torch.zeros(B, T_t, dtype=torch.bool),
        "alignment_student_chunk_id": torch.full((B, T_s), -1, dtype=torch.long),
        "alignment_teacher_chunk_id": torch.full((B, T_t), -1, dtype=torch.long),
        "alignment_num_chunks": torch.tensor([1, 2], dtype=torch.long),
        # extra unrelated keys must be ignored
        "input_ids": torch.zeros(B, T_s),
    }
    batch = lu.alignment_from_flat_batch(data)
    assert isinstance(batch, AlignmentBatch)
    assert batch.pair_valid.shape == (B, P)
    assert batch.student_chunk_id.shape == (B, T_s)
    assert batch.num_chunks.tolist() == [1, 2]
