"""CPU tests for the multi-teacher CrossTokenizerDistillationLossFn.

The loss is exercised single-rank / non-DTensor by passing synthetic per-teacher
``teacher_full_logits_by_idx`` + ``aligns_by_idx`` straight in (bypassing the
transport). Covers the single-teacher (num_teachers==1) P-KL and gold paths,
reverse-KL, dynamic loss scaling, the empty-valid-chunk early return, the
(False gold, True xtoken) reject, sample_mask gating, and the multi-teacher
surface: two cross-tokenizer teachers (weighted sum), a same-tokenizer teacher
(direct KL, no projection), the averaged_logits / select_teacher modes, dynamic
teacher weighting, per-teacher ``_t{i}`` metrics, and the config validators.
"""

from typing import Any, cast

import torch

from dockyard_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn
from dockyard_rl.algorithms.x_token.loss_utils import LocalizedAlignment
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

VS, VT, K = 8, 6, 4


def _projection_file(tmp_path, name="proj.pt"):
    # Dense top-k projection: student row i -> teacher (i % VT) with weight 1.0,
    # second slot -1 (the strict exact-map sentinel). Exercises both the common
    # (exact-mapped) and uncommon partitions in the gold path.
    idx = torch.tensor([[i % VT, -1] for i in range(VS)])
    lik = torch.tensor([[1.0, 0.0] for _ in range(VS)])
    p = tmp_path / name
    torch.save({"indices": idx, "likelihoods": lik}, p)
    return str(p)


def _cfg(paths, *, vts=None, weights=None, gold=None, xtoken=None, **over):
    """Multi-teacher loss config. ``paths[i] is None`` marks a same-tokenizer teacher."""
    n = len(paths)
    cfg = dict(
        gold_loss=False, xtoken_loss=False,
        temperature=1.0, vocab_topk=K, uncommon_topk=8192, reverse_kl=False,
        exact_token_match_only=False, kl_loss_weight=1.0, ce_loss_scale=1.0,
        dynamic_loss_scaling=False,
        kd_loss_mode="sum", normalize_teacher_by_vocab=False, alpha=1.0,
        student_vocab_size=VS,
        projection_matrix_paths=list(paths),
        teacher_vocab_sizes=list(vts if vts is not None else [VT] * n),
        teacher_weights=list(weights if weights is not None else [1.0] * n),
        teacher_gold_loss=list(gold if gold is not None else [None] * n),
        teacher_xtoken_loss=list(xtoken if xtoken is not None else [None] * n),
    )
    cfg.update(over)
    return cfg


def _single(path, **over):
    return _cfg([path], **over)


def _data(b=1, t=4, sample_mask=None):
    bd: BatchedDataDict = BatchedDataDict()
    bd["input_ids"] = torch.randint(0, VS, (b, t))
    bd["token_mask"] = torch.ones(b, t)
    bd["sample_mask"] = torch.ones(b) if sample_mask is None else sample_mask
    return bd


def _align(b=1, t=4, n_chunks=3, sample_mask=None):
    """Cross-tokenizer alignment: chunk/pair fields populated."""
    chunk = [list(range(n_chunks)) + [-1] * (t - n_chunks)]
    return LocalizedAlignment(
        sample_mask=torch.ones(b) if sample_mask is None else sample_mask,
        student_chunk_id=torch.tensor(chunk * b),
        teacher_chunk_id=torch.tensor(chunk * b),
        pair_valid=torch.ones(b, n_chunks, dtype=torch.bool),
        pair_is_correct=torch.ones(b, n_chunks, dtype=torch.bool),
        student_input_ids=torch.randint(0, VS, (b, t)),
        student_token_mask=torch.ones(b, t),
    )


def _same_align(b=1, t=4, sample_mask=None):
    """Same-tokenizer (thin) alignment: only the shared student fields."""
    return LocalizedAlignment(
        sample_mask=torch.ones(b) if sample_mask is None else sample_mask,
        student_input_ids=torch.randint(0, VS, (b, t)),
        student_token_mask=torch.ones(b, t),
    )


def _run(loss_fn, teacher_by_idx, aligns_by_idx, data, *, b=1, t=4, requires_grad=True):
    logits = torch.randn(b, t, VS, requires_grad=requires_grad)
    return (
        loss_fn(
            data, torch.tensor(3.0), torch.tensor(3.0),
            logits, logits, teacher_by_idx, aligns_by_idx,
        ),
        logits,
    )


# -- single-teacher P-KL ------------------------------------------------------

def test_pkl_finite_loss_grad_and_metrics(tmp_path):
    fn = CrossTokenizerDistillationLossFn(_single(_projection_file(tmp_path)))
    (loss, metrics), logits = _run(fn, {0: torch.randn(1, 4, VT)}, {0: _align()}, _data())
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert {"loss", "kl_loss", "ce_loss", "kl_loss_scale", "accuracy",
            "num_valid_samples", "kl_loss_t0", "proj_accuracy_t0",
            "num_valid_pairs_t0", "weight_t0"} <= set(metrics)
    assert metrics["num_valid_pairs_t0"] == 3  # 3 valid chunks, 1 unmasked sample
    assert metrics["weight_t0"] == 1.0


def _pkl_loss_and_grad(tmp_path, *, project, seed=7, **cfg_over):
    """Run the single-teacher P-KL path with a chosen projection implementation."""
    from dockyard_rl.algorithms.loss import loss_functions as lf

    torch.manual_seed(seed)
    logits = torch.randn(1, 4, VS, requires_grad=True)
    teacher = torch.randn(1, 4, VT)
    data = _data()
    fn = CrossTokenizerDistillationLossFn(
        cast(Any, _single(_projection_file(tmp_path), **cfg_over))
    )

    original = lf.project_student_to_teacher_vocab
    lf.project_student_to_teacher_vocab = project
    try:
        loss, _ = fn(
            data, torch.tensor(3.0), torch.tensor(3.0),
            logits, logits, {0: teacher}, {0: _align()},
        )
        loss.backward()
    finally:
        lf.project_student_to_teacher_vocab = original
    assert logits.grad is not None
    return loss.detach().clone(), logits.grad.clone()


def _project_then_index(student_probs, sparse_projection, *, tp_group=None,
                        teacher_columns=None):
    """The pre-#3564 route: build the full [B, T, V_t] projection, then index it."""
    from dockyard_rl.algorithms.x_token.loss_utils import (
        project_student_to_teacher_vocab as real,
    )

    full = real(student_probs, sparse_projection, tp_group=tp_group)
    return full if teacher_columns is None else full[..., teacher_columns]


def test_pkl_topk_projection_matches_projecting_the_full_vocab(tmp_path):
    """#3564 is a memory change, so loss and gradient must be unmoved.

    Projecting into the top-k teacher columns replaces materializing
    [B, T, V_t] fp32 -- all-reduced across TP -- and reading k of its columns.
    """
    from dockyard_rl.algorithms.x_token.loss_utils import (
        project_student_to_teacher_vocab as real,
    )

    ref_loss, ref_grad = _pkl_loss_and_grad(tmp_path, project=_project_then_index)
    got_loss, got_grad = _pkl_loss_and_grad(tmp_path, project=real)

    torch.testing.assert_close(got_loss, ref_loss)
    torch.testing.assert_close(got_grad, ref_grad)


def test_pkl_matches_when_topk_covers_the_whole_teacher_vocab(tmp_path):
    """The degenerate k == V_t case still routes through the column slice."""
    from dockyard_rl.algorithms.x_token.loss_utils import (
        project_student_to_teacher_vocab as real,
    )

    ref_loss, ref_grad = _pkl_loss_and_grad(
        tmp_path, project=_project_then_index, vocab_topk=VT
    )
    got_loss, got_grad = _pkl_loss_and_grad(tmp_path, project=real, vocab_topk=VT)

    torch.testing.assert_close(got_loss, ref_loss)
    torch.testing.assert_close(got_grad, ref_grad)


def test_pkl_never_builds_the_full_vocab_projection(tmp_path):
    """The wire: the projection call must carry the top-k columns.

    Goes red if _compute_p_kl reverts to projecting the full teacher vocab and
    indexing the result -- the equivalence tests above cannot catch that, since
    both routes give the same numbers.
    """
    from dockyard_rl.algorithms.loss import loss_functions as lf
    from dockyard_rl.algorithms.x_token.loss_utils import (
        project_student_to_teacher_vocab as real,
    )

    seen: list[tuple[torch.Tensor | None, int]] = []

    def spy(student_probs, sparse_projection, *, tp_group=None, teacher_columns=None):
        out = real(
            student_probs,
            sparse_projection,
            tp_group=tp_group,
            teacher_columns=teacher_columns,
        )
        seen.append((teacher_columns, int(out.shape[-1])))
        return out

    assert K < VT, "the test only means something when top-k is a strict subset"
    _pkl_loss_and_grad(tmp_path, project=spy)

    assert seen, "the P-KL path did not call the projection"
    for columns, width in seen:
        assert columns is not None, "projection was called without teacher_columns"
        assert int(columns.numel()) == K
        assert width == K, f"projected width {width} is the full teacher vocab"


def test_pkl_reverse_kl_runs(tmp_path):
    fn = CrossTokenizerDistillationLossFn(_single(_projection_file(tmp_path), reverse_kl=True))
    (loss, _), _ = _run(fn, {0: torch.randn(1, 4, VT)}, {0: _align()}, _data())
    assert torch.isfinite(loss)


def test_pkl_dynamic_loss_scaling_sets_scale(tmp_path):
    fn = CrossTokenizerDistillationLossFn(
        _single(_projection_file(tmp_path), dynamic_loss_scaling=True)
    )
    (loss, metrics), _ = _run(fn, {0: torch.randn(1, 4, VT)}, {0: _align()}, _data())
    assert torch.isfinite(loss)
    # dynamic scale = |ce| / |kd| (not the fixed 1.0).
    assert metrics["kl_loss_scale"] != 1.0


# -- single-teacher gold ------------------------------------------------------

def test_gold_finite_loss_and_metrics(tmp_path):
    fn = CrossTokenizerDistillationLossFn(_single(_projection_file(tmp_path), gold_loss=True))
    (loss, metrics), logits = _run(fn, {0: torch.randn(1, 4, VT)}, {0: _align()}, _data())
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert {"loss", "ce_loss", "accuracy", "kl_loss_t0", "kl_common_t0",
            "l1_uncommon_t0", "num_valid_chunks_t0"} <= set(metrics)
    assert metrics["num_valid_chunks_t0"] == 3


def test_gold_xtoken_modifier_runs(tmp_path):
    fn = CrossTokenizerDistillationLossFn(
        _single(_projection_file(tmp_path), gold_loss=True, xtoken_loss=True)
    )
    (loss, _), _ = _run(fn, {0: torch.randn(1, 4, VT)}, {0: _align()}, _data())
    assert torch.isfinite(loss)


# -- guards / edge cases ------------------------------------------------------

def test_xtoken_loss_requires_gold_loss(tmp_path):
    try:
        CrossTokenizerDistillationLossFn(
            _single(_projection_file(tmp_path), gold_loss=False, xtoken_loss=True)
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_all_samples_masked_gives_zero_loss_no_nan(tmp_path):
    fn = CrossTokenizerDistillationLossFn(_single(_projection_file(tmp_path)))
    (loss, metrics), _ = _run(
        fn, {0: torch.randn(1, 4, VT)},
        {0: _align(sample_mask=torch.tensor([0.0]))},
        _data(sample_mask=torch.tensor([0.0])),
    )
    # empty-valid-chunk early return: KL term is zero; CE is masked out too.
    assert torch.isfinite(loss)
    assert metrics["num_valid_pairs_t0"] == 0


def test_sample_mask_gates_a_masked_row_out_of_kl(tmp_path):
    # A sample_mask=0 row must not contribute to the KL term. With two samples
    # where the second is masked, num_valid_pairs counts only sample 0's chunks.
    fn = CrossTokenizerDistillationLossFn(_single(_projection_file(tmp_path)))
    sm = torch.tensor([1.0, 0.0])
    (loss, metrics), _ = _run(
        fn, {0: torch.randn(2, 4, VT)}, {0: _align(b=2, sample_mask=sm)},
        _data(b=2, sample_mask=sm), b=2,
    )
    assert torch.isfinite(loss)
    assert metrics["num_valid_pairs_t0"] == 3  # only sample 0's 3 chunks count


# -- multi-teacher ------------------------------------------------------------

def test_two_cross_tokenizer_teachers_weighted_sum(tmp_path):
    p0 = _projection_file(tmp_path, "p0.pt")
    p1 = _projection_file(tmp_path, "p1.pt")
    fn = CrossTokenizerDistillationLossFn(_cfg([p0, p1], weights=[1.0, 0.5]))
    (loss, metrics), _ = _run(
        fn, {0: torch.randn(1, 4, VT), 1: torch.randn(1, 4, VT)},
        {0: _align(), 1: _align()}, _data(),
    )
    assert torch.isfinite(loss)
    assert fn.num_teachers == 2
    # per-teacher metrics for both teachers + the aggregate.
    assert {"kl_loss_t0", "kl_loss_t1", "weight_t0", "weight_t1"} <= set(metrics)
    assert metrics["weight_t0"] == 1.0 and metrics["weight_t1"] == 0.5
    # aggregate KD equals w0*kd0 + w1*kd1.
    expected = 1.0 * metrics["kl_loss_t0"] + 0.5 * metrics["kl_loss_t1"]
    assert abs(metrics["kl_loss"] - expected) < 1e-4


def test_same_tokenizer_teacher_direct_kl(tmp_path):
    # projection None -> same-tokenizer teacher: direct top-k KL, no alignment.
    fn = CrossTokenizerDistillationLossFn(_cfg([None], vts=[VS]))
    (loss, metrics), logits = _run(
        fn, {0: torch.randn(1, 4, VS)}, {0: _same_align()}, _data()
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert "kl_loss_t0" in metrics


def _topk_kl_loss_and_grad(*, gather, seed=11):
    """Run the same-tokenizer direct top-k KL with a chosen student-gather."""
    from dockyard_rl.algorithms.loss import loss_functions as lf

    torch.manual_seed(seed)
    logits = torch.randn(1, 4, VS, requires_grad=True)
    teacher = torch.randn(1, 4, VS)
    fn = CrossTokenizerDistillationLossFn(cast(Any, _cfg([None], vts=[VS])))

    original = lf.vocab_parallel_gather_columns
    lf.vocab_parallel_gather_columns = gather
    try:
        loss, _ = fn(
            _data(), torch.tensor(3.0), torch.tensor(3.0),
            logits, logits, {0: teacher}, {0: _same_align()},
        )
        loss.backward()
    finally:
        lf.vocab_parallel_gather_columns = original
    assert logits.grad is not None
    return loss.detach().clone(), logits.grad.clone()


def test_direct_topk_kl_matches_the_full_vocab_log_softmax_route(tmp_path):
    """#3314 is a memory change; loss and gradient must be unmoved.

    The reference reproduces the pre-change code exactly: full-vocab
    log-softmax, slice to the top-k columns, renormalize within the subset.
    Returning raw logits instead is only valid because the full-vocab partition
    function cancels in that renormalization.
    """
    from dockyard_rl.distributed.model_utils import (
        vocab_parallel_full_log_softmax,
        vocab_parallel_gather_columns,
    )

    def reference(logits, columns, *, tp_group=None):
        # The old route, expressed so its output feeds the same log_softmax the
        # new call site applies: renormalized subset log-probs are a valid set
        # of "logits" up to the constant that log_softmax removes.
        log_probs = vocab_parallel_full_log_softmax(logits, 1.0, tp_group=tp_group)
        gathered = log_probs[..., columns]
        return gathered - torch.logsumexp(gathered, dim=-1, keepdim=True)

    ref_loss, ref_grad = _topk_kl_loss_and_grad(gather=reference)
    got_loss, got_grad = _topk_kl_loss_and_grad(gather=vocab_parallel_gather_columns)

    torch.testing.assert_close(got_loss, ref_loss)
    torch.testing.assert_close(got_grad, ref_grad)


def test_averaged_logits_two_same_tokenizer(tmp_path):
    fn = CrossTokenizerDistillationLossFn(
        _cfg([None, None], vts=[VS, VS], weights=[0.7, 0.3], kd_loss_mode="averaged_logits")
    )
    (loss, _), _ = _run(
        fn, {0: torch.randn(1, 4, VS), 1: torch.randn(1, 4, VS)},
        {0: _same_align(), 1: _same_align()}, _data(),
    )
    assert torch.isfinite(loss)


def test_select_teacher_picks_one(tmp_path):
    fn = CrossTokenizerDistillationLossFn(
        _cfg([None, None], vts=[VS, VS], kd_loss_mode="select_teacher")
    )
    (loss, metrics), _ = _run(
        fn, {0: torch.randn(1, 4, VS), 1: torch.randn(1, 4, VS)},
        {0: _same_align(), 1: _same_align()}, _data(),
    )
    assert torch.isfinite(loss)
    assert metrics["selected_teacher"] in (0, 1)


def test_dynamic_ce_weights_form_a_softmax(tmp_path):
    fn = CrossTokenizerDistillationLossFn(
        _cfg([None, None], vts=[VS, VS], kd_loss_mode="sum", sum_weights_metric="ce")
    )
    (loss, metrics), _ = _run(
        fn, {0: torch.randn(1, 4, VS), 1: torch.randn(1, 4, VS)},
        {0: _same_align(), 1: _same_align()}, _data(),
    )
    assert torch.isfinite(loss)
    # softmax weights across teachers sum to 1.
    assert abs((metrics["weight_t0"] + metrics["weight_t1"]) - 1.0) < 1e-5


# -- multi-teacher config validators ------------------------------------------

def test_sum_weights_metric_rejected_outside_sum_mode(tmp_path):
    try:
        CrossTokenizerDistillationLossFn(
            _cfg([None, None], vts=[VS, VS], kd_loss_mode="select_teacher",
                 sum_weights_metric="ce")
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_averaged_logits_zero_weight_sum_rejected(tmp_path):
    try:
        CrossTokenizerDistillationLossFn(
            _cfg([None, None], vts=[VS, VS], weights=[1.0, -1.0],
                 kd_loss_mode="averaged_logits")
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_unequal_per_teacher_list_lengths_rejected(tmp_path):
    cfg = _cfg([None, None], vts=[VS, VS])
    cfg["teacher_weights"] = [1.0]  # length 1 vs 2 teachers
    try:
        CrossTokenizerDistillationLossFn(cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass
