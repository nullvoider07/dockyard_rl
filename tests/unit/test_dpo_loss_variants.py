"""Unit tests for the preference-loss variant factory and the variants
(DPO, cDPO, DPOP, R-DPO, IPO, KTO) plus the unpaired KTO collate.

Pure CPU coverage. The math is validated three ways:
  1. Identity reduction — each paired variant equals vanilla DPO at its identity
     hyperparameter (cDPO ε=0, DPOP λ=0, R-DPO α=0), byte-identical loss + metrics.
  2. Closed-form parity — an independent reimplementation of each variant's
     reward/margin spec (incl. the KTO value function) matches the loss-function
     output on tiny fixtures.
  3. Hand-computed scalar — a single fully hand-derived value guards against a
     systematic error shared by the loss and its reimplementation.

The KTO collate is covered with unpaired-batching shape tests. These fixtures
double as the JAX phase-J3 byte-parity contract (Bucket B).
"""

import math
from typing import cast

import pytest
import torch

from dockyard_rl.algorithms.loss import DPOLossConfig

from dockyard_rl.algorithms.loss import (
    CDPOLossFn,
    DPOLossFn,
    DPOPLossFn,
    IPOLossFn,
    KTOLossFn,
    ORPOLossFn,
    RDPOLossFn,
    SimPOLossFn,
    build_preference_loss,
    is_reference_free,
)
from transformers import PreTrainedTokenizerBase
from dockyard_rl.data.collate_fn import kto_collate_fn, KTODatumSpec
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict


BETA = 0.1


def _base_cfg(**overrides) -> DPOLossConfig:
    cfg = {
        "reference_policy_kl_penalty": BETA,
        "preference_loss_weight": 1.0,
        "sft_loss_weight": 0.0,
        "preference_average_log_probs": False,
        "sft_average_log_probs": False,
    }
    cfg.update(overrides)
    return cast(DPOLossConfig, cfg)


def _fixture():
    """Two preference pairs (interleaved chosen/rejected → batch of 4).

    next_token_logprobs: [B, L]; token_mask / ref logprobs: [B, L+1] (the loss
    slices token_mask[:, 1:] and ref[:, :-1]).
    """
    B = 4
    ntlp = torch.tensor(
        [
            [-0.5, -1.0, -0.2],   # pair0 chosen
            [-1.5, -2.0, -0.7],   # pair0 rejected
            [-0.3, -0.9, -1.1],   # pair1 chosen
            [-2.0, -1.2, -0.4],   # pair1 rejected
        ]
    )
    ref = torch.tensor(
        [
            [0.0, -0.6, -1.3, -0.9],
            [0.0, -1.0, -1.8, -1.0],
            [0.0, -0.4, -0.8, -1.6],
            [0.0, -0.5, -1.4, -0.3],
        ]
    )
    token_mask = torch.tensor(
        [
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],   # shorter chosen → exercise length-norm
            [0.0, 1.0, 1.0, 1.0],
        ]
    )
    sample_mask = torch.ones(B)
    data = BatchedDataDict(
        {
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "reference_policy_logprobs": ref,
        }
    )
    gvs = torch.tensor(float(B))
    gvt = token_mask[:, 1:].sum()
    return ntlp, data, gvs, gvt


def _seq_rewards(ntlp, data, average_log_probs):
    """Independent reimplementation of the per-sequence reward s_x."""
    tm = data["token_mask"][:, 1:]
    ref = data["reference_policy_logprobs"][:, :-1]
    s = ((ntlp - ref) * tm).sum(-1)
    if average_log_probs:
        s = s / tm.sum(-1).clamp(min=1)
    return s


def _ref_dpo_loss(ntlp, data, gvs, beta, average_log_probs=False,
                  label_smoothing=0.0, dpop_lambda=0.0, length_penalty=0.0):
    """Closed-form reference: DPO / cDPO / DPOP / R-DPO from the spec."""
    s = _seq_rewards(ntlp, data, average_log_probs)
    s_c, s_r = s[::2], s[1::2]
    delta = s_c - s_r

    if dpop_lambda > 0.0:
        tm = data["token_mask"][:, 1:]
        ref = data["reference_policy_logprobs"][:, :-1]
        policy_sum = (ntlp * tm).sum(-1)
        ref_sum = (ref * tm).sum(-1)
        if average_log_probs:
            denom = tm.sum(-1).clamp(min=1)
            policy_sum = policy_sum / denom
            ref_sum = ref_sum / denom
        penalty_c = torch.clamp(ref_sum - policy_sum, min=0.0)[::2]
        delta = delta - dpop_lambda * penalty_c

    logits = beta * delta
    if length_penalty > 0.0:
        # α length penalty is applied outside the β scaling (R-DPO).
        lengths = data["token_mask"][:, 1:].sum(-1)
        delta_len = lengths[::2] - lengths[1::2]
        logits = logits - length_penalty * delta_len

    if label_smoothing > 0.0:
        per_pair = -(
            (1.0 - label_smoothing) * torch.nn.functional.logsigmoid(logits)
            + label_smoothing * torch.nn.functional.logsigmoid(-logits)
        )
    else:
        per_pair = -torch.nn.functional.logsigmoid(logits)
    # masked_mean with all-valid sample_mask[::2] and global factor gvs/2.
    return (per_pair * data["sample_mask"][::2]).sum() / (gvs / 2)


def _ref_ipo_loss(ntlp, data, gvs, beta, average_log_probs=False):
    """Closed-form reference: IPO squared loss (Δ − 1/(2β))²."""
    s = _seq_rewards(ntlp, data, average_log_probs)
    delta = s[::2] - s[1::2]
    target = 1.0 / (2.0 * beta)
    per_pair = (delta - target) ** 2
    return (per_pair * data["sample_mask"][::2]).sum() / (gvs / 2)


# factory + registry
class TestFactory:
    def test_builds_each_variant(self):
        assert isinstance(build_preference_loss("dpo", _base_cfg()), DPOLossFn)
        assert isinstance(
            build_preference_loss("dpop", _base_cfg(dpop_lambda=0.5)), DPOPLossFn
        )
        assert isinstance(
            build_preference_loss("cdpo", _base_cfg(label_smoothing=0.1)), CDPOLossFn
        )
        assert isinstance(
            build_preference_loss("rdpo", _base_cfg(length_penalty=0.05)), RDPOLossFn
        )
        assert isinstance(build_preference_loss("ipo", _base_cfg()), IPOLossFn)
        assert isinstance(build_preference_loss("kto", _base_cfg()), KTOLossFn)
        assert isinstance(build_preference_loss("simpo", _base_cfg()), SimPOLossFn)
        assert isinstance(build_preference_loss("orpo", _base_cfg()), ORPOLossFn)

    def test_case_insensitive(self):
        assert isinstance(build_preference_loss("DPOP", _base_cfg()), DPOPLossFn)
        assert isinstance(build_preference_loss("IPO", _base_cfg()), IPOLossFn)
        assert isinstance(build_preference_loss("KTO", _base_cfg()), KTOLossFn)
        assert isinstance(build_preference_loss("SimPO", _base_cfg()), SimPOLossFn)

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="loss_variant"):
            build_preference_loss("foobar", _base_cfg())

    def test_none_variant_defaults_to_dpo(self):
        # config `loss_variant: null` (key present, value None) must not crash.
        assert type(build_preference_loss(None, _base_cfg())) is DPOLossFn
        assert is_reference_free(None) is False

    def test_reference_using_variants(self):
        for variant in ("dpo", "dpop", "cdpo", "rdpo", "ipo", "kto"):
            assert is_reference_free(variant) is False

    def test_reference_free_variants(self):
        # SimPO/ORPO are the only reference-free variants → the guard flips here.
        assert is_reference_free("simpo") is True
        assert is_reference_free("orpo") is True

    def test_unknown_variant_not_reference_free(self):
        assert is_reference_free("foobar") is False

    def test_variant_hyperparameter_bounds(self):
        with pytest.raises(AssertionError):
            CDPOLossFn(_base_cfg(label_smoothing=0.5))
        with pytest.raises(AssertionError):
            DPOPLossFn(_base_cfg(dpop_lambda=-0.1))
        with pytest.raises(AssertionError):
            RDPOLossFn(_base_cfg(length_penalty=-0.1))
        with pytest.raises(AssertionError):
            KTOLossFn(_base_cfg(desirable_weight=-0.1))
        with pytest.raises(AssertionError):
            SimPOLossFn(_base_cfg(simpo_gamma=-0.1))
        with pytest.raises(AssertionError):
            ORPOLossFn(_base_cfg(orpo_lambda=-0.1))

    def test_sequences_per_datum_contract(self):
        # Paired family declares 2; unpaired KTO declares 1 (driver reads this).
        assert DPOLossFn(_base_cfg()).sequences_per_datum == 2
        assert DPOPLossFn(_base_cfg()).sequences_per_datum == 2
        assert SimPOLossFn(_base_cfg()).sequences_per_datum == 2
        assert ORPOLossFn(_base_cfg()).sequences_per_datum == 2
        assert KTOLossFn(_base_cfg()).sequences_per_datum == 1


# identity reductions
class TestIdentityReduction:
    def test_cdpo_eps0_equals_dpo(self):
        ntlp, data, gvs, gvt = _fixture()
        dpo = DPOLossFn(_base_cfg())
        cdpo = CDPOLossFn(_base_cfg(label_smoothing=0.0))
        l_dpo, m_dpo = dpo(ntlp, data, gvs, gvt)
        l_cdpo, m_cdpo = cdpo(ntlp, data, gvs, gvt)
        assert torch.equal(l_dpo, l_cdpo)
        assert m_dpo["preference_loss"] == m_cdpo["preference_loss"]
        assert m_dpo["accuracy"] == m_cdpo["accuracy"]

    def test_dpop_lambda0_equals_dpo(self):
        ntlp, data, gvs, gvt = _fixture()
        dpo = DPOLossFn(_base_cfg())
        dpop = DPOPLossFn(_base_cfg(dpop_lambda=0.0))
        l_dpo, m_dpo = dpo(ntlp, data, gvs, gvt)
        l_dpop, m_dpop = dpop(ntlp, data, gvs, gvt)
        assert torch.equal(l_dpo, l_dpop)
        assert m_dpo["preference_loss"] == m_dpop["preference_loss"]

    def test_dpop_lambda0_equals_dpo_with_length_norm(self):
        ntlp, data, gvs, gvt = _fixture()
        cfg = _base_cfg(preference_average_log_probs=True)
        l_dpo, _ = DPOLossFn(cfg)(ntlp, data, gvs, gvt)
        l_dpop, _ = DPOPLossFn(cast(DPOLossConfig, cfg | {"dpop_lambda": 0.0}))(ntlp, data, gvs, gvt)
        assert torch.equal(l_dpo, l_dpop)

    def test_rdpo_alpha0_equals_dpo(self):
        ntlp, data, gvs, gvt = _fixture()
        dpo = DPOLossFn(_base_cfg())
        rdpo = RDPOLossFn(_base_cfg(length_penalty=0.0))
        l_dpo, m_dpo = dpo(ntlp, data, gvs, gvt)
        l_rdpo, m_rdpo = rdpo(ntlp, data, gvs, gvt)
        assert torch.equal(l_dpo, l_rdpo)
        assert m_dpo["preference_loss"] == m_rdpo["preference_loss"]


# closed-form parity
class TestClosedFormParity:
    @pytest.mark.parametrize("average_log_probs", [False, True])
    def test_dpo_matches_spec(self, average_log_probs):
        ntlp, data, gvs, gvt = _fixture()
        cfg = _base_cfg(preference_average_log_probs=average_log_probs)
        loss, _ = DPOLossFn(cfg)(ntlp, data, gvs, gvt)
        expected = _ref_dpo_loss(ntlp, data, gvs, BETA, average_log_probs)
        torch.testing.assert_close(loss, expected)

    @pytest.mark.parametrize("eps", [0.05, 0.1, 0.25])
    def test_cdpo_matches_spec(self, eps):
        ntlp, data, gvs, gvt = _fixture()
        loss, _ = CDPOLossFn(_base_cfg(label_smoothing=eps))(ntlp, data, gvs, gvt)
        expected = _ref_dpo_loss(ntlp, data, gvs, BETA, label_smoothing=eps)
        torch.testing.assert_close(loss, expected)

    @pytest.mark.parametrize("lam", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("average_log_probs", [False, True])
    def test_dpop_matches_spec(self, lam, average_log_probs):
        ntlp, data, gvs, gvt = _fixture()
        cfg = _base_cfg(dpop_lambda=lam, preference_average_log_probs=average_log_probs)
        loss, _ = DPOPLossFn(cfg)(ntlp, data, gvs, gvt)
        expected = _ref_dpo_loss(
            ntlp, data, gvs, BETA, average_log_probs=average_log_probs, dpop_lambda=lam
        )
        torch.testing.assert_close(loss, expected)

    def test_dpop_penalty_active_only_when_chosen_below_ref(self):
        """DPOP must differ from DPO exactly when a chosen seq drops below ref."""
        ntlp, data, gvs, gvt = _fixture()
        l_dpo, _ = DPOLossFn(_base_cfg())(ntlp, data, gvs, gvt)
        l_dpop, _ = DPOPLossFn(_base_cfg(dpop_lambda=1.0))(ntlp, data, gvs, gvt)
        # Penalty subtracts from the margin → loss strictly increases when any
        # chosen sequence has s_ref_chosen > s_policy_chosen.
        ref = data["reference_policy_logprobs"][:, :-1]
        tm = data["token_mask"][:, 1:]
        penalty = torch.clamp(((ref - ntlp) * tm).sum(-1), min=0.0)[::2]
        if (penalty > 0).any():
            assert l_dpop.item() > l_dpo.item()
        else:
            assert l_dpop.item() == pytest.approx(l_dpo.item())

    @pytest.mark.parametrize("alpha", [0.05, 0.2, 0.5])
    @pytest.mark.parametrize("average_log_probs", [False, True])
    def test_rdpo_matches_spec(self, alpha, average_log_probs):
        ntlp, data, gvs, gvt = _fixture()
        cfg = _base_cfg(length_penalty=alpha, preference_average_log_probs=average_log_probs)
        loss, _ = RDPOLossFn(cfg)(ntlp, data, gvs, gvt)
        expected = _ref_dpo_loss(
            ntlp, data, gvs, BETA, average_log_probs=average_log_probs, length_penalty=alpha
        )
        torch.testing.assert_close(loss, expected)

    @pytest.mark.parametrize("average_log_probs", [False, True])
    def test_ipo_matches_spec(self, average_log_probs):
        ntlp, data, gvs, gvt = _fixture()
        cfg = _base_cfg(preference_average_log_probs=average_log_probs)
        loss, _ = IPOLossFn(cfg)(ntlp, data, gvs, gvt)
        expected = _ref_ipo_loss(ntlp, data, gvs, BETA, average_log_probs)
        torch.testing.assert_close(loss, expected)

    def test_ipo_is_not_dpo(self):
        """IPO is a distinct family — it must not coincide with DPO here."""
        ntlp, data, gvs, gvt = _fixture()
        l_dpo, _ = DPOLossFn(_base_cfg())(ntlp, data, gvs, gvt)
        l_ipo, _ = IPOLossFn(_base_cfg())(ntlp, data, gvs, gvt)
        assert not torch.allclose(l_dpo, l_ipo)


# hand-computed scalar guard
class TestHandComputed:
    def test_single_pair_dpo_exact(self):
        """One pair, β=1, no length-norm — fully hand-derived loss."""
        # The loss slices token_mask[:, 1:] and ref[:, :-1]; ntlp aligns with both.
        # chosen: ntlp=[-1.0], ref slice=[-1.5] → s_c = -1.0 - (-1.5) = 0.5
        # rejected: ntlp=[-2.0], ref slice=[-1.0] → s_r = -2.0 - (-1.0) = -1.0
        # Δ = 1.5 ; L = -logσ(1.5) = softplus(-1.5)
        ntlp = torch.tensor([[-1.0], [-2.0]])
        ref = torch.tensor([[-1.5, 0.0], [-1.0, 0.0]])
        token_mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        data = BatchedDataDict(
            {
                "token_mask": token_mask,
                "sample_mask": torch.ones(2),
                "reference_policy_logprobs": ref,
            }
        )
        gvs = torch.tensor(2.0)
        gvt = token_mask[:, 1:].sum()
        loss, _ = DPOLossFn(_base_cfg(reference_policy_kl_penalty=1.0))(
            ntlp, data, gvs, gvt
        )
        expected = math.log1p(math.exp(-1.5))  # softplus(-1.5) = -logσ(1.5)
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_single_pair_ipo_exact(self):
        """Same single-pair fixture: IPO target 1/(2β), β=0.5 → target=1.

        Δ = 1.5 (s_c=0.5, s_r=-1.0); L = (1.5 − 1.0)² = 0.25.
        """
        ntlp = torch.tensor([[-1.0], [-2.0]])
        ref = torch.tensor([[-1.5, 0.0], [-1.0, 0.0]])
        token_mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        data = BatchedDataDict(
            {
                "token_mask": token_mask,
                "sample_mask": torch.ones(2),
                "reference_policy_logprobs": ref,
            }
        )
        gvs = torch.tensor(2.0)
        gvt = token_mask[:, 1:].sum()
        loss, _ = IPOLossFn(_base_cfg(reference_policy_kl_penalty=0.5))(
            ntlp, data, gvs, gvt
        )
        assert loss.item() == pytest.approx(0.25, abs=1e-6)

    def test_single_pair_rdpo_exact(self):
        """Single pair, β=1, α=0.5 — chosen len 1, rejected len 2 (penalty active).

        s_c = -1.0-(-1.5) = 0.5; s_r = (-2.0+1.0)+(-0.5+1.0) = -0.5; Δ = 1.0.
        Δlen = |y_c| − |y_r| = 1 − 2 = −1.
        margin = β·Δ − α·Δlen = 1.0 − 0.5·(−1) = 1.5; L = softplus(−1.5).
        """
        ntlp = torch.tensor([[-1.0, 0.0], [-2.0, -0.5]])
        ref = torch.tensor([[-1.5, 0.0, 0.0], [-1.0, -1.0, 0.0]])
        token_mask = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        data = BatchedDataDict(
            {
                "token_mask": token_mask,
                "sample_mask": torch.ones(2),
                "reference_policy_logprobs": ref,
            }
        )
        gvs = torch.tensor(2.0)
        gvt = token_mask[:, 1:].sum()
        loss, _ = RDPOLossFn(
            _base_cfg(reference_policy_kl_penalty=1.0, length_penalty=0.5)
        )(ntlp, data, gvs, gvt)
        expected = math.log1p(math.exp(-1.5))
        assert loss.item() == pytest.approx(expected, abs=1e-6)


# KTO (unpaired)
def _kto_fixture():
    """Four independently-labelled examples (2 desirable, 2 undesirable)."""
    ntlp = torch.tensor(
        [
            [-0.5, -1.0, -0.2],   # desirable
            [-1.5, -2.0, -0.7],   # undesirable
            [-0.3, -0.9, -1.1],   # desirable (shorter)
            [-2.0, -1.2, -0.4],   # undesirable
        ]
    )
    ref = torch.tensor(
        [
            [0.0, -0.6, -1.3, -0.9],
            [0.0, -1.0, -1.8, -1.0],
            [0.0, -0.4, -0.8, -1.6],
            [0.0, -0.5, -1.4, -0.3],
        ]
    )
    token_mask = torch.tensor(
        [
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
    )
    label = torch.tensor([1.0, 0.0, 1.0, 0.0])
    sample_mask = torch.ones(4)
    data = BatchedDataDict(
        {
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "reference_policy_logprobs": ref,
            "preference_label": label,
        }
    )
    gvs = torch.tensor(4.0)
    gvt = token_mask[:, 1:].sum()
    return ntlp, data, gvs, gvt


def _ref_kto_loss(ntlp, data, gvs, beta, kl, desirable_weight, undesirable_weight):
    """Closed-form reference for the KTO unpaired value function."""
    tm = data["token_mask"][:, 1:]
    ref = data["reference_policy_logprobs"][:, :-1]
    r = ((ntlp - ref) * tm).sum(-1)
    label = data["preference_label"]
    kl_t = torch.as_tensor(kl, dtype=r.dtype)
    des = 1.0 - torch.sigmoid(beta * (r - kl_t))
    und = 1.0 - torch.sigmoid(beta * (kl_t - r))
    per = label * desirable_weight * des + (1.0 - label) * undesirable_weight * und
    return (per * data["sample_mask"]).sum() / gvs


class TestKTOLoss:
    @pytest.mark.parametrize("kl", [0.0, 0.3, -0.2])
    @pytest.mark.parametrize("dw,uw", [(1.0, 1.0), (1.0, 1.33), (2.0, 0.5)])
    def test_kto_matches_spec(self, kl, dw, uw):
        ntlp, data, gvs, gvt = _kto_fixture()
        if kl != 0.0:
            data["kto_reference_kl"] = torch.tensor(kl)
        loss, _ = KTOLossFn(
            _base_cfg(desirable_weight=dw, undesirable_weight=uw)
        )(ntlp, data, gvs, gvt)
        expected = _ref_kto_loss(ntlp, data, gvs, BETA, kl, dw, uw)
        torch.testing.assert_close(loss, expected)

    def test_kto_kl_absent_defaults_zero(self):
        """No kto_reference_kl in data → z=0 (pre-worker behaviour)."""
        ntlp, data, gvs, gvt = _kto_fixture()
        assert "kto_reference_kl" not in data
        loss, metrics = KTOLossFn(_base_cfg())(ntlp, data, gvs, gvt)
        expected = _ref_kto_loss(ntlp, data, gvs, BETA, 0.0, 1.0, 1.0)
        torch.testing.assert_close(loss, expected)
        assert metrics["kl"] == 0.0

    def test_kto_label_selects_branch(self):
        """All-desirable vs all-undesirable use different loss forms."""
        ntlp, data, gvs, gvt = _kto_fixture()
        data_des = BatchedDataDict(dict(data))
        data_des["preference_label"] = torch.ones(4)
        data_und = BatchedDataDict(dict(data))
        data_und["preference_label"] = torch.zeros(4)
        loss_des, m_des = KTOLossFn(_base_cfg())(ntlp, data_des, gvs, gvt)
        loss_und, m_und = KTOLossFn(_base_cfg())(ntlp, data_und, gvs, gvt)
        assert m_des["num_desirable"] == 4 and m_des["num_undesirable"] == 0
        assert m_und["num_desirable"] == 0 and m_und["num_undesirable"] == 4
        torch.testing.assert_close(
            loss_des, _ref_kto_loss(ntlp, data_des, gvs, BETA, 0.0, 1.0, 1.0)
        )
        torch.testing.assert_close(
            loss_und, _ref_kto_loss(ntlp, data_und, gvs, BETA, 0.0, 1.0, 1.0)
        )

    def test_kto_weight_scales_loss(self):
        """On an all-desirable batch, doubling λ_D doubles the loss."""
        ntlp, data, gvs, gvt = _kto_fixture()
        data["preference_label"] = torch.ones(4)
        l1, _ = KTOLossFn(_base_cfg(desirable_weight=1.0))(ntlp, data, gvs, gvt)
        l2, _ = KTOLossFn(_base_cfg(desirable_weight=2.0))(ntlp, data, gvs, gvt)
        torch.testing.assert_close(l2, 2.0 * l1)

    def test_single_desirable_kto_exact(self):
        """One desirable example, β=1, z=0: r=0.5 → L = 1 − σ(0.5)."""
        ntlp = torch.tensor([[-1.0]])
        ref = torch.tensor([[-1.5, 0.0]])
        token_mask = torch.tensor([[0.0, 1.0]])
        data = BatchedDataDict(
            {
                "token_mask": token_mask,
                "sample_mask": torch.ones(1),
                "reference_policy_logprobs": ref,
                "preference_label": torch.tensor([1.0]),
            }
        )
        gvs = torch.tensor(1.0)
        gvt = token_mask[:, 1:].sum()
        loss, _ = KTOLossFn(_base_cfg(reference_policy_kl_penalty=1.0))(
            ntlp, data, gvs, gvt
        )
        expected = 1.0 - 1.0 / (1.0 + math.exp(-0.5))
        assert loss.item() == pytest.approx(expected, abs=1e-6)


# KTO collate (unpaired)
class _FakeTokenizer:
    pad_token_id = 0


def _kto_datum(idx, label, prompt_ids, completion_ids, loss_multiplier=1.0):
    return {
        "message_log": [
            {"role": "user", "token_ids": torch.tensor(prompt_ids)},
            {"role": "assistant", "token_ids": torch.tensor(completion_ids)},
        ],
        "length": len(prompt_ids) + len(completion_ids),
        "preference_label": label,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }


class TestKTOCollate:
    def test_batch_size_not_doubled(self):
        batch = [
            _kto_datum(0, True, [1, 2, 3], [4, 5]),
            _kto_datum(1, False, [6, 7], [8, 9, 10]),
            _kto_datum(2, True, [11], [12, 13]),
        ]
        data = kto_collate_fn(
            cast(list[KTODatumSpec], batch), tokenizer=cast(PreTrainedTokenizerBase, _FakeTokenizer()),
            make_sequence_length_divisible_by=1, add_loss_mask=True,
        )
        # Unpaired: one row per datum (preference_collate_fn would give 2×).
        assert data["input_ids"].shape[0] == 3
        assert data["preference_label"].shape == (3,)

    def test_preference_label_values(self):
        batch = [
            _kto_datum(0, True, [1, 2], [3]),
            _kto_datum(1, False, [4], [5, 6]),
            _kto_datum(2, True, [7, 8], [9]),
        ]
        data = kto_collate_fn(
            cast(list[KTODatumSpec], batch), tokenizer=cast(PreTrainedTokenizerBase, _FakeTokenizer()),
            make_sequence_length_divisible_by=1, add_loss_mask=True,
        )
        assert torch.equal(data["preference_label"], torch.tensor([1.0, 0.0, 1.0]))

    def test_token_mask_unmasks_completion_only(self):
        # only_unmask_final=True → only the assistant turn is unmasked.
        batch = [_kto_datum(0, True, [1, 2, 3], [4, 5])]
        data = kto_collate_fn(
            cast(list[KTODatumSpec], batch), tokenizer=cast(PreTrainedTokenizerBase, _FakeTokenizer()),
            make_sequence_length_divisible_by=1, add_loss_mask=True,
        )
        # 2 completion tokens unmasked out of 5 total.
        assert data["token_mask"].sum().item() == 2.0
        assert "sample_mask" in data

    def test_sample_mask_from_loss_multiplier(self):
        batch = [
            _kto_datum(0, True, [1, 2], [3], loss_multiplier=1.0),
            _kto_datum(1, False, [4], [5], loss_multiplier=0.0),
        ]
        data = kto_collate_fn(
            cast(list[KTODatumSpec], batch), tokenizer=cast(PreTrainedTokenizerBase, _FakeTokenizer()),
            make_sequence_length_divisible_by=1, add_loss_mask=True,
        )
        assert torch.equal(data["sample_mask"], torch.tensor([1.0, 0.0]))


# reference-free variants (SimPO, ORPO)
def _reffree_fixture():
    """Two pairs, no reference_policy_logprobs (proves reference-free).

    Average log-probs are all < −log2 so ORPO's log1mexp impl and the test's
    independent reimplementation hit the same numerical branch (exact parity).
    """
    ntlp = torch.tensor(
        [
            [-1.0, -1.2, -0.9],   # pair0 chosen   avg -1.033
            [-1.5, -1.1, -1.3],   # pair0 rejected avg -1.300
            [-0.8, -1.4, -1.0],   # pair1 chosen   (2 tokens) avg -1.100
            [-1.6, -0.9, -1.2],   # pair1 rejected avg -1.233
        ]
    )
    token_mask = torch.tensor(
        [
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
    )
    sample_mask = torch.ones(4)
    data = BatchedDataDict(
        {
            "token_mask": token_mask,
            "sample_mask": sample_mask,
        }
    )
    gvs = torch.tensor(4.0)
    gvt = token_mask[:, 1:].sum()
    return ntlp, data, gvs, gvt


def _avg_logp(ntlp, data):
    tm = data["token_mask"][:, 1:]
    return (ntlp * tm).sum(-1) / tm.sum(-1).clamp(min=1)


def _ref_simpo_loss(ntlp, data, gvs, beta, gamma):
    s = _avg_logp(ntlp, data)
    logits = beta * (s[::2] - s[1::2]) - gamma
    per_pair = -torch.nn.functional.logsigmoid(logits)
    return (per_pair * data["sample_mask"][::2]).sum() / (gvs / 2)


def _ref_orpo_loss(ntlp, data, gvs, lam):
    s = _avg_logp(ntlp, data)
    s_c, s_r = s[::2], s[1::2]
    log1mexp = lambda x: torch.log1p(-torch.exp(x))  # noqa: E731 (x < −log2 here)
    log_odds = (s_c - s_r) - (log1mexp(s_c) - log1mexp(s_r))
    or_loss = -torch.nn.functional.logsigmoid(log_odds)
    per_pair = -s_c + lam * or_loss
    return (per_pair * data["sample_mask"][::2]).sum() / (gvs / 2)


class TestSimPO:
    def test_no_reference_logprobs_needed(self):
        # Fixture omits reference_policy_logprobs entirely.
        ntlp, data, gvs, gvt = _reffree_fixture()
        assert "reference_policy_logprobs" not in data
        SimPOLossFn(_base_cfg())(ntlp, data, gvs, gvt)  # must not raise

    @pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
    def test_simpo_matches_spec(self, gamma):
        ntlp, data, gvs, gvt = _reffree_fixture()
        loss, _ = SimPOLossFn(_base_cfg(simpo_gamma=gamma))(ntlp, data, gvs, gvt)
        expected = _ref_simpo_loss(ntlp, data, gvs, BETA, gamma)
        torch.testing.assert_close(loss, expected)

    def test_gamma_increases_loss(self):
        ntlp, data, gvs, gvt = _reffree_fixture()
        l0, _ = SimPOLossFn(_base_cfg(simpo_gamma=0.0))(ntlp, data, gvs, gvt)
        l1, _ = SimPOLossFn(_base_cfg(simpo_gamma=1.0))(ntlp, data, gvs, gvt)
        # A positive target margin subtracts from every pair's logit → larger loss.
        assert l1.item() > l0.item()

    def test_single_pair_simpo_exact(self):
        """β=1, γ=0.5; s_c=-1.0, s_r=-2.0 → logit=1.0-0.5=0.5 → L=softplus(-0.5)."""
        ntlp = torch.tensor([[-1.0], [-2.0]])
        token_mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        data = BatchedDataDict(
            {"token_mask": token_mask, "sample_mask": torch.ones(2)}
        )
        gvs = torch.tensor(2.0)
        loss, _ = SimPOLossFn(
            _base_cfg(reference_policy_kl_penalty=1.0, simpo_gamma=0.5)
        )(ntlp, data, gvs, token_mask[:, 1:].sum())
        expected = math.log1p(math.exp(-0.5))
        assert loss.item() == pytest.approx(expected, abs=1e-6)


class TestORPO:
    def test_no_reference_logprobs_needed(self):
        ntlp, data, gvs, gvt = _reffree_fixture()
        assert "reference_policy_logprobs" not in data
        ORPOLossFn(_base_cfg())(ntlp, data, gvs, gvt)  # must not raise

    @pytest.mark.parametrize("lam", [0.1, 0.5, 1.0])
    def test_orpo_matches_spec(self, lam):
        ntlp, data, gvs, gvt = _reffree_fixture()
        loss, _ = ORPOLossFn(_base_cfg(orpo_lambda=lam))(ntlp, data, gvs, gvt)
        expected = _ref_orpo_loss(ntlp, data, gvs, lam)
        torch.testing.assert_close(loss, expected)

    def test_orpo_metrics_present(self):
        ntlp, data, gvs, gvt = _reffree_fixture()
        _, metrics = ORPOLossFn(_base_cfg(orpo_lambda=0.5))(ntlp, data, gvs, gvt)
        for key in ("sft_loss", "or_loss", "log_odds_ratio", "accuracy"):
            assert key in metrics

    def test_log1mexp_matches_naive_below_half(self):
        # For x < −log2, the stable branch equals log1p(−exp(x)).
        from dockyard_rl.algorithms.loss.loss_functions import _log1mexp
        x = torch.tensor([-0.8, -1.0, -2.5, -5.0])
        torch.testing.assert_close(_log1mexp(x), torch.log1p(-torch.exp(x)))

    def test_log1mexp_matches_naive_above_half(self):
        # For x > −log2, the stable branch equals log(−expm1(x)).
        from dockyard_rl.algorithms.loss.loss_functions import _log1mexp
        x = torch.tensor([-0.1, -0.3, -0.5])
        torch.testing.assert_close(_log1mexp(x), torch.log(-torch.expm1(x)))
