"""CPU tests for the prepare_loss_input dispatcher (M2.a).

The dispatcher turns raw student logits into the per-input-type kwargs each loss
fn consumes. These cover the non-parallel CPU paths of LOGIT / LOGPROB /
DISTILLATION routing, the DRAFT NotImplementedError, the unknown-type
ValueError, and that the DISTILLATION_CROSS_TOKENIZER branch is reached (its
keystone is M3-deferred, so it raises ImportError until then).
"""

from types import SimpleNamespace

import pytest
import torch

from dockyard_rl.algorithms.loss.interfaces import LossInputType
from dockyard_rl.algorithms.loss.utils import prepare_loss_input
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict


def _data(**kw):
    bd: BatchedDataDict = BatchedDataDict()
    for k, v in kw.items():
        bd[k] = v
    return bd


def _loss_fn(input_type, **attrs):
    # A loss fn stand-in: prepare_loss_input only reads .input_type plus a few
    # optional attributes via hasattr.
    return SimpleNamespace(input_type=input_type, **attrs)


# -- LOGIT --------------------------------------------------------------------

def test_logit_passes_logits_through():
    logits = torch.randn(2, 5, 7)
    loss_input, _ = prepare_loss_input(
        logits, _data(input_ids=torch.zeros(2, 5, dtype=torch.long)),
        _loss_fn(LossInputType.LOGIT),
    )
    assert set(loss_input) == {"logits"}
    assert loss_input["logits"] is logits


# -- LOGPROB ------------------------------------------------------------------

def test_logprob_routes_and_forwards_kwargs(monkeypatch):
    # The non-parallel logprob helper hardcodes .cuda() (model_utils.py), so it
    # is GPU-only; here we monkeypatch it (the dispatcher imports it function-
    # locally, so patching the source module is picked up at call time) and
    # assert the dispatcher routes to it and forwards the expected kwargs.
    import dockyard_rl.distributed.model_utils as mu

    captured = {}
    sentinel = torch.randn(2, 5)

    def fake(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(mu, "get_next_token_logprobs_from_logits", fake)
    logits = torch.randn(2, 6, 11)
    input_ids = torch.randint(0, 11, (2, 6))
    loss_input, _ = prepare_loss_input(
        logits, _data(input_ids=input_ids), _loss_fn(LossInputType.LOGPROB),
    )
    assert loss_input["next_token_logprobs"] is sentinel
    assert captured["next_token_logits"] is logits
    assert captured["input_ids"] is input_ids
    assert captured["sampling_params"] is None


def test_logprob_linear_ce_fusion_uses_precomputed_logprobs():
    b, t = 2, 6
    precomputed = torch.randn(b, t - 1)  # already next-token logprobs
    loss_input, _ = prepare_loss_input(
        precomputed,
        _data(input_ids=torch.zeros(b, t, dtype=torch.long)),
        _loss_fn(LossInputType.LOGPROB, use_linear_ce_fusion=True),
    )
    lp = loss_input["next_token_logprobs"]
    assert lp.shape == (b, t - 1)
    assert torch.allclose(lp, precomputed.float())


def test_logprob_no_filtering_does_not_write_unfiltered(monkeypatch):
    import dockyard_rl.distributed.model_utils as mu

    calls = {"n": 0}

    def fake(**kwargs):
        calls["n"] += 1
        return torch.randn(1, 3)

    monkeypatch.setattr(mu, "get_next_token_logprobs_from_logits", fake)
    b, t, v = 1, 4, 5
    data = _data(input_ids=torch.randint(0, v, (b, t)))
    prepare_loss_input(
        torch.randn(b, t, v), data,
        # reference_policy_kl_penalty set, but sampling_params=None so the
        # filtering block is skipped and curr_logprobs_unfiltered is never set
        # (the helper is called exactly once, not a second time for unfiltered).
        _loss_fn(LossInputType.LOGPROB, reference_policy_kl_penalty=0.1),
        sampling_params=None,
    )
    assert "curr_logprobs_unfiltered" not in data
    assert calls["n"] == 1


# -- DISTILLATION -------------------------------------------------------------

def test_distillation_routes_and_packs_triplet(monkeypatch):
    # get_distillation_topk_logprobs_from_logits is device-coupled (GPU-only);
    # monkeypatch it and assert the dispatcher forwards kwargs and packs the
    # returned triplet into the loss_input dict.
    import dockyard_rl.distributed.model_utils as mu

    captured = {}
    triplet = (torch.randn(2, 5, 4), torch.randn(2, 5, 4), torch.randn(2, 5))

    def fake(**kwargs):
        captured.update(kwargs)
        return triplet

    monkeypatch.setattr(mu, "get_distillation_topk_logprobs_from_logits", fake)
    logits = torch.randn(2, 5, 13)
    loss_input, _ = prepare_loss_input(
        logits,
        _data(
            input_ids=torch.randint(0, 13, (2, 5)),
            teacher_topk_logits=torch.randn(2, 5, 4),
            teacher_topk_indices=torch.randint(0, 13, (2, 5, 4)),
        ),
        _loss_fn(LossInputType.DISTILLATION, zero_outside_topk=True, kl_type="reverse"),
    )
    assert set(loss_input) == {"student_topk_logprobs", "teacher_topk_logprobs", "H_all"}
    assert loss_input["student_topk_logprobs"] is triplet[0]
    assert loss_input["H_all"] is triplet[2]
    assert captured["student_logits"] is logits
    # zero_outside_topk + non-forward kl => calculate_entropy True.
    assert captured["calculate_entropy"] is True


def test_distillation_forward_kl_sets_calculate_entropy_false(monkeypatch):
    import dockyard_rl.distributed.model_utils as mu

    captured = {}
    monkeypatch.setattr(
        mu,
        "get_distillation_topk_logprobs_from_logits",
        lambda **kw: (captured.update(kw) or (torch.zeros(1), torch.zeros(1), None)),
    )
    prepare_loss_input(
        torch.randn(1, 4, 9),
        _data(
            input_ids=torch.randint(0, 9, (1, 4)),
            teacher_topk_logits=torch.randn(1, 4, 3),
            teacher_topk_indices=torch.randint(0, 9, (1, 4, 3)),
        ),
        _loss_fn(LossInputType.DISTILLATION, zero_outside_topk=True, kl_type="forward"),
    )
    # forward kl => calculate_entropy False regardless of zero_outside_topk.
    assert captured["calculate_entropy"] is False


# -- DRAFT / unknown ----------------------------------------------------------

def test_draft_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="DRAFT"):
        prepare_loss_input(
            torch.randn(1, 3, 4),
            _data(input_ids=torch.zeros(1, 3, dtype=torch.long)),
            _loss_fn(LossInputType.DRAFT),
        )


def test_unknown_input_type_raises_value_error():
    with pytest.raises(ValueError, match="Unknown loss function input type"):
        prepare_loss_input(
            torch.randn(1, 3, 4),
            _data(input_ids=torch.zeros(1, 3, dtype=torch.long)),
            _loss_fn("not_a_real_type"),
        )


# -- DISTILLATION_CROSS_TOKENIZER (branch reached; keystone M3-deferred) -------

def test_cross_tokenizer_branch_reaches_deferred_keystone():
    # The branch lazy-imports prepare_xtoken_cross_tokenizer_loss_input, which is
    # not built until M3, so reaching the branch raises ImportError. This pins
    # that the new enum routes to the xtoken path (not the unknown-type error).
    with pytest.raises(ImportError):
        prepare_loss_input(
            torch.randn(1, 3, 4),
            _data(input_ids=torch.zeros(1, 3, dtype=torch.long)),
            _loss_fn(LossInputType.DISTILLATION_CROSS_TOKENIZER),
        )
