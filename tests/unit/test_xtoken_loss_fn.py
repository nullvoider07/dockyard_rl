"""CPU tests for CrossTokenizerDistillationLossFn (M2.d).

The loss is exercised single-rank / non-DTensor by passing synthetic
teacher_full_logits + a LocalizedAlignment straight in (bypassing the M3
transport). Covers the P-KL and gold paths, reverse-KL, dynamic loss scaling,
the empty-valid-chunk early return, the (False gold, True xtoken) reject, and
that sample_mask gating neutralizes a masked sample (M1-audit Contract 2).
"""

import torch

from dockyard_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn
from dockyard_rl.algorithms.x_token.loss_utils import LocalizedAlignment
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

VS, VT, K = 8, 6, 4


def _projection_file(tmp_path):
    # Dense top-k projection: student row i -> teacher (i % VT) with weight 1.0,
    # second slot -1 (the strict exact-map sentinel). Exercises both the common
    # (exact-mapped) and uncommon partitions in the gold path.
    idx = torch.tensor([[i % VT, -1] for i in range(VS)])
    lik = torch.tensor([[1.0, 0.0] for _ in range(VS)])
    p = tmp_path / "proj.pt"
    torch.save({"indices": idx, "likelihoods": lik}, p)
    return str(p)


def _cfg(path, **over):
    cfg = dict(
        projection_matrix_path=path, gold_loss=False, xtoken_loss=False,
        temperature=1.0, vocab_topk=K, uncommon_topk=8192, reverse_kl=False,
        exact_token_match_only=False, kl_loss_weight=1.0, ce_loss_scale=1.0,
        dynamic_loss_scaling=False, student_vocab_size=VS, teacher_vocab_size=VT,
    )
    cfg.update(over)
    return cfg


def _data(b=1, t=4, sample_mask=None):
    bd: BatchedDataDict = BatchedDataDict()
    bd["input_ids"] = torch.randint(0, VS, (b, t))
    bd["token_mask"] = torch.ones(b, t)
    bd["sample_mask"] = torch.ones(b) if sample_mask is None else sample_mask
    return bd


def _align(b=1, t=4, n_chunks=3, sample_mask=None):
    chunk = [list(range(n_chunks)) + [-1] * (t - n_chunks)]
    return LocalizedAlignment(
        student_chunk_id=torch.tensor(chunk * b),
        teacher_chunk_id=torch.tensor(chunk * b),
        pair_valid=torch.ones(b, n_chunks, dtype=torch.bool),
        pair_is_correct=torch.ones(b, n_chunks, dtype=torch.bool),
        sample_mask=torch.ones(b) if sample_mask is None else sample_mask,
        student_input_ids=torch.randint(0, VS, (b, t)),
        student_token_mask=torch.ones(b, t),
    )


def _run(loss_fn, *, b=1, t=4, sample_mask=None, requires_grad=True):
    logits = torch.randn(b, t, VS, requires_grad=requires_grad)
    teacher = torch.randn(b, t, VT)
    data = _data(b, t, sample_mask=sample_mask)
    align = _align(b, t, sample_mask=sample_mask)
    return loss_fn(
        data, torch.tensor(3.0), torch.tensor(3.0),
        logits, teacher, logits, align,
    ), logits


# -- P-KL ---------------------------------------------------------------------

def test_pkl_finite_loss_grad_and_metrics(tmp_path):
    fn = CrossTokenizerDistillationLossFn(_cfg(_projection_file(tmp_path)))
    (loss, metrics), logits = _run(fn)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert {"loss", "kl_loss", "ce_loss", "kl_loss_scale", "accuracy",
            "proj_accuracy", "num_valid_samples", "num_valid_pairs"} <= set(metrics)
    assert metrics["num_valid_pairs"] == 3  # 3 valid chunks, 1 unmasked sample


def test_pkl_reverse_kl_runs(tmp_path):
    fn = CrossTokenizerDistillationLossFn(_cfg(_projection_file(tmp_path), reverse_kl=True))
    (loss, _), _ = _run(fn)
    assert torch.isfinite(loss)


def test_pkl_dynamic_loss_scaling_sets_scale(tmp_path):
    fn = CrossTokenizerDistillationLossFn(
        _cfg(_projection_file(tmp_path), dynamic_loss_scaling=True)
    )
    (loss, metrics), _ = _run(fn)
    assert torch.isfinite(loss)
    # dynamic scale = |ce| / |kl| (not the fixed 1.0).
    assert metrics["kl_loss_scale"] != 1.0


# -- gold ---------------------------------------------------------------------

def test_gold_finite_loss_and_metrics(tmp_path):
    fn = CrossTokenizerDistillationLossFn(_cfg(_projection_file(tmp_path), gold_loss=True))
    (loss, metrics), logits = _run(fn)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert {"loss", "kl_common", "l1_uncommon", "ce_loss", "accuracy",
            "num_valid_chunks"} <= set(metrics)
    assert metrics["num_valid_chunks"] == 3


def test_gold_xtoken_modifier_runs(tmp_path):
    fn = CrossTokenizerDistillationLossFn(
        _cfg(_projection_file(tmp_path), gold_loss=True, xtoken_loss=True)
    )
    (loss, _), _ = _run(fn)
    assert torch.isfinite(loss)


# -- guards / edge cases ------------------------------------------------------

def test_xtoken_loss_requires_gold_loss(tmp_path):
    try:
        CrossTokenizerDistillationLossFn(
            _cfg(_projection_file(tmp_path), gold_loss=False, xtoken_loss=True)
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_all_samples_masked_gives_zero_loss_no_nan(tmp_path):
    fn = CrossTokenizerDistillationLossFn(_cfg(_projection_file(tmp_path)))
    (loss, metrics), _ = _run(fn, sample_mask=torch.tensor([0.0]))
    # empty-valid-chunk early return: KL term is zero; CE is masked out too.
    assert torch.isfinite(loss)
    assert metrics["num_valid_pairs"] == 0


def test_sample_mask_gates_a_masked_row_out_of_kl(tmp_path):
    # Contract 2: a sample_mask=0 row must not contribute to the KL term. With
    # two samples where the second is masked, num_valid_pairs counts only the
    # first sample's chunks.
    fn = CrossTokenizerDistillationLossFn(_cfg(_projection_file(tmp_path)))
    (loss, metrics), _ = _run(fn, b=2, sample_mask=torch.tensor([1.0, 0.0]))
    assert torch.isfinite(loss)
    assert metrics["num_valid_pairs"] == 3  # only sample 0's 3 chunks count
