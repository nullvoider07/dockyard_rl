"""Unit tests for the online/iterative DPO core (D5).

Pure CPU coverage of the genuinely-new online-preference logic: the preference
judge (+ verifiable fallback), best-vs-worst pair construction with the margin
gate, pair→preference-datum conversion (validated through the real
preference_collate_fn), the over-optimization monitor, and the per-iteration /
loop drivers (exercised with stub generation/judge/train callables).

Live generation/judge/train throughput is GPU-deferred (HV); here the loop is
driven entirely by injected stubs.
"""

import pytest
import torch

from dockyard_rl.algorithms.online_dpo import (
    Candidate,
    JudgeOveroptMonitor,
    OnlineDPOConfig,
    PreferenceJudge,
    build_preference_pairs,
    online_dpo_train,
    pairs_to_preference_data,
    parse_judge_score,
    run_online_dpo_iteration,
)
from dockyard_rl.data.collate_fn import preference_collate_fn
from dockyard_rl.rewards.interfaces import RewardVerificationResult


# fakes
class _FakeJudgeClient:
    """Minimal HLEJudgeClient stand-in: scores by response length proxy."""

    def __init__(self, enabled=True, reply="SCORE: 80"):
        self._enabled = enabled
        self._reply = reply
        self.calls = 0

    @property
    def enabled(self):
        return self._enabled

    def chat(self, messages, *, temperature=0):
        self.calls += 1
        return self._reply


class _FakeTokenizer:
    pad_token_id = 0


def _cand(score=None, prompt_ids=(1, 2), resp_ids=(3, 4), text="resp"):
    return Candidate(
        prompt_token_ids=torch.tensor(prompt_ids),
        response_token_ids=torch.tensor(resp_ids),
        prompt_text="prompt",
        response_text=text,
        score=score,
    )


# score parsing
class TestParseScore:
    def test_basic(self):
        assert parse_judge_score("SCORE: 80") == pytest.approx(0.8)

    def test_clamps_above_100(self):
        assert parse_judge_score("SCORE: 250") == 1.0

    def test_float(self):
        assert parse_judge_score("blah\nSCORE: 12.5\nok") == pytest.approx(0.125)

    def test_no_match(self):
        assert parse_judge_score("no number here") is None

    def test_none_input(self):
        assert parse_judge_score(None) is None


# preference judge + verifiable fallback
class TestPreferenceJudge:
    def test_judge_scoring(self):
        judge = PreferenceJudge(_FakeJudgeClient(reply="SCORE: 60"))
        assert judge.enabled
        assert judge.score("p", "r") == pytest.approx(0.6)

    def test_disabled_when_nothing_configured(self):
        assert PreferenceJudge(None).enabled is False
        assert PreferenceJudge(_FakeJudgeClient(enabled=False)).enabled is False

    def test_verifiable_preferred_over_judge(self):
        client = _FakeJudgeClient(reply="SCORE: 10")
        verifiable = lambda traj: RewardVerificationResult(  # noqa: E731
            reward=1.0, status="ok", evidence_hash=None, failure_reason=None
        )
        judge = PreferenceJudge(client, verifiable_reward=verifiable, prefer_verifiable=True)
        # Verifiable ground truth wins; judge not even called.
        assert judge.score("p", "r", trajectory={"x": 1}) == 1.0
        assert client.calls == 0

    def test_verifiable_falls_back_to_judge_when_not_ok(self):
        client = _FakeJudgeClient(reply="SCORE: 30")
        verifiable = lambda traj: RewardVerificationResult(  # noqa: E731
            reward=0.0, status="execution_error", evidence_hash=None, failure_reason="x"
        )
        judge = PreferenceJudge(client, verifiable_reward=verifiable, prefer_verifiable=True)
        # Verifiable could not score (not "ok") → judge is used.
        assert judge.score("p", "r", trajectory={"x": 1}) == pytest.approx(0.3)
        assert client.calls == 1

    def test_no_trajectory_uses_judge(self):
        client = _FakeJudgeClient(reply="SCORE: 45")
        verifiable = lambda traj: RewardVerificationResult(  # noqa: E731
            reward=1.0, status="ok", evidence_hash=None, failure_reason=None
        )
        judge = PreferenceJudge(client, verifiable_reward=verifiable)
        # No trajectory → verifiable inapplicable → judge.
        assert judge.score("p", "r", trajectory=None) == pytest.approx(0.45)

    def test_prefer_judge_ordering(self):
        client = _FakeJudgeClient(reply="SCORE: 70")
        verifiable = lambda traj: RewardVerificationResult(  # noqa: E731
            reward=1.0, status="ok", evidence_hash=None, failure_reason=None
        )
        judge = PreferenceJudge(client, verifiable_reward=verifiable, prefer_verifiable=False)
        # Judge first; verifiable only as last resort.
        assert judge.score("p", "r", trajectory={"x": 1}) == pytest.approx(0.7)

    def test_returns_none_when_judge_fails(self):
        judge = PreferenceJudge(_FakeJudgeClient(reply="garbage"))
        assert judge.score("p", "r") is None


# pair construction (best-vs-worst + margin gate)
class TestBuildPairs:
    def test_best_vs_worst(self):
        groups = [[_cand(0.2), _cand(0.9), _cand(0.5)]]
        pairs, metrics = build_preference_pairs(groups, score_margin=0.1)
        assert len(pairs) == 1
        chosen, rejected = pairs[0]
        assert chosen.score == 0.9 and rejected.score == 0.2
        assert metrics["num_pairs"] == 1

    def test_margin_gate_drops_near_ties(self):
        groups = [[_cand(0.50), _cand(0.55)]]  # gap 0.05 < margin
        pairs, metrics = build_preference_pairs(groups, score_margin=0.1)
        assert pairs == []
        assert metrics["num_dropped_margin"] == 1

    def test_margin_gate_keeps_above(self):
        groups = [[_cand(0.3), _cand(0.9)]]  # gap 0.6 ≫ margin
        pairs, _ = build_preference_pairs(groups, score_margin=0.1)
        assert len(pairs) == 1

    def test_unscored_group_dropped(self):
        groups = [[_cand(None), _cand(0.8)], [_cand(0.1), _cand(0.9)]]
        pairs, metrics = build_preference_pairs(groups, score_margin=0.1)
        assert len(pairs) == 1  # first group has <2 scored
        assert metrics["num_dropped_unscored"] == 1

    def test_metrics_counts(self):
        groups = [
            [_cand(0.1), _cand(0.9)],   # kept
            [_cand(0.5), _cand(0.52)],  # margin drop
            [_cand(None)],              # unscored drop
        ]
        _, metrics = build_preference_pairs(groups, score_margin=0.1)
        assert metrics == {
            "num_groups": 3,
            "num_pairs": 1,
            "num_dropped_unscored": 1,
            "num_dropped_margin": 1,
        }


# pair → preference data → collate
class TestPairsToData:
    def test_datum_spec_fields(self):
        pairs = [(_cand(0.9, resp_ids=(3, 4, 5)), _cand(0.1, resp_ids=(6,)))]
        data = pairs_to_preference_data(pairs, start_idx=7)
        d = data[0]
        assert d["idx"] == 7
        assert d["loss_multiplier"] == 1.0
        assert d["length_chosen"] == 2 + 3   # prompt 2 + response 3
        assert d["length_rejected"] == 2 + 1
        assert d["message_log_chosen"][1]["role"] == "assistant"

    def test_flows_through_preference_collate(self):
        # The online path must produce exactly the paired batch offline DPO eats.
        pairs = [
            (_cand(0.9, resp_ids=(3, 4)), _cand(0.1, resp_ids=(5,))),
            (_cand(0.8, resp_ids=(6,)), _cand(0.2, resp_ids=(7, 8))),
        ]
        data = pairs_to_preference_data(pairs)
        from typing import cast
        from transformers import PreTrainedTokenizerBase
        batch = preference_collate_fn(
            data, tokenizer=cast(PreTrainedTokenizerBase, _FakeTokenizer()),
            make_sequence_length_divisible_by=1, add_loss_mask=True,
        )
        # 2 pairs interleaved chosen/rejected → batch of 4.
        assert batch["input_ids"].shape[0] == 4
        assert "token_mask" in batch and "sample_mask" in batch


# over-optimization monitor
class TestMonitor:
    def test_running_mean(self):
        m = JudgeOveroptMonitor()
        m.update(0.4)
        out = m.update(0.6)
        assert out["judge_score_running_mean"] == pytest.approx(0.5)

    def test_kl_over_target_flag(self):
        m = JudgeOveroptMonitor(kl_target=0.1)
        assert m.update(0.5, kl=0.05)["kl_over_target"] is False
        assert m.update(0.5, kl=0.2)["kl_over_target"] is True

    def test_no_kl_target_no_flag(self):
        m = JudgeOveroptMonitor(kl_target=None)
        assert "kl_over_target" not in m.update(0.5, kl=0.2)


# iteration + loop wiring (stub generate/judge/train)
def _stub_generate(scores_per_prompt):
    """Return a generate_fn producing candidates with preset scores."""
    def gen(prompts):
        groups = []
        for scores in scores_per_prompt:
            groups.append([_cand(s) for s in scores])
        return groups
    return gen


class TestIteration:
    def test_full_iteration_trains(self):
        calls = {}

        def train_fn(batch):
            calls["batch"] = batch
            return {"loss": 0.5, "kl": 0.03}

        metrics = run_online_dpo_iteration(
            ["p0", "p1"],
            generate_fn=_stub_generate([[0.1, 0.9], [0.2, 0.8]]),
            judge=PreferenceJudge(_FakeJudgeClient()),  # scores preset → judge unused
            score_margin=0.1,
            build_batch_fn=lambda data: ("batch", len(data)),
            train_fn=train_fn,
            monitor=JudgeOveroptMonitor(kl_target=0.1),
        )
        assert metrics["trained"] is True
        assert metrics["num_pairs"] == 2
        assert metrics["train/loss"] == 0.5
        assert metrics["monitor/kl_over_target"] is False
        assert calls["batch"] == ("batch", 2)

    def test_iteration_skips_when_no_pairs(self):
        train_called = {"n": 0}

        def train_fn(batch):
            train_called["n"] += 1
            return {}

        metrics = run_online_dpo_iteration(
            ["p0"],
            generate_fn=_stub_generate([[0.50, 0.52]]),  # below margin → dropped
            judge=PreferenceJudge(_FakeJudgeClient()),
            score_margin=0.1,
            build_batch_fn=lambda data: data,
            train_fn=train_fn,
        )
        assert metrics["trained"] is False
        assert metrics["num_pairs"] == 0
        assert train_called["n"] == 0  # no DPO step taken

    def test_judge_scores_unscored_candidates(self):
        # Candidates arrive without scores → the judge fills them in.
        def gen(prompts):
            return [[_cand(None, text="a"), _cand(None, text="b")]]

        # Judge returns alternating scores via a stateful client.
        class _AltClient:
            enabled = True
            def __init__(self):
                self.seq = iter(["SCORE: 90", "SCORE: 10"])
            def chat(self, messages, *, temperature=0):
                return next(self.seq)

        metrics = run_online_dpo_iteration(
            ["p"],
            generate_fn=gen,
            judge=PreferenceJudge(_AltClient()),
            score_margin=0.1,
            build_batch_fn=lambda data: data,
            train_fn=lambda batch: {"loss": 0.1},
            monitor=JudgeOveroptMonitor(),
        )
        assert metrics["num_pairs"] == 1
        assert metrics["judge_score_mean"] == pytest.approx(0.5)


class TestLoop:
    def _config(self, **over) -> OnlineDPOConfig:
        cfg = {
            "candidates_per_prompt": 2,
            "score_margin": 0.1,
            "prefer_verifiable": True,
            "regeneration_period": 1,
            "max_num_steps": 10,
        }
        cfg.update(over)
        return cfg  # type: ignore[return-value]

    def test_loop_runs_all_batches(self):
        steps = []

        def train_fn(batch):
            steps.append(1)
            return {"loss": 0.2, "kl": 0.01}

        metrics = online_dpo_train(
            [["p0"], ["p1"], ["p2"]],
            generate_fn=_stub_generate([[0.1, 0.9]]),
            judge=PreferenceJudge(_FakeJudgeClient()),
            build_batch_fn=lambda data: data,
            train_fn=train_fn,
            config=self._config(),
        )
        assert len(metrics) == 3
        assert sum(steps) == 3
        assert [m["step"] for m in metrics] == [0, 1, 2]

    def test_loop_honors_max_steps(self):
        metrics = online_dpo_train(
            [["p"]] * 5,
            generate_fn=_stub_generate([[0.1, 0.9]]),
            judge=PreferenceJudge(_FakeJudgeClient()),
            build_batch_fn=lambda data: data,
            train_fn=lambda batch: {"loss": 0.1},
            config=self._config(max_num_steps=2),
        )
        assert len(metrics) == 2

    def test_loop_auto_monitor_from_kl_target(self):
        metrics = online_dpo_train(
            [["p"]],
            generate_fn=_stub_generate([[0.1, 0.9]]),
            judge=PreferenceJudge(_FakeJudgeClient()),
            build_batch_fn=lambda data: data,
            train_fn=lambda batch: {"loss": 0.1, "kl": 0.5},
            config=self._config(kl_target=0.1),
        )
        # kl_target set → monitor auto-created → flag surfaces in metrics.
        assert metrics[0]["monitor/kl_over_target"] is True

    def test_loop_idx_accumulates_across_steps(self):
        seen_idx = []

        def build_batch_fn(data):
            seen_idx.append(data[0]["idx"])
            return data

        online_dpo_train(
            [["p"], ["p"]],
            generate_fn=_stub_generate([[0.1, 0.9]]),
            judge=PreferenceJudge(_FakeJudgeClient()),
            build_batch_fn=build_batch_fn,
            train_fn=lambda batch: {"loss": 0.1},
            config=self._config(),
        )
        # Second step's first datum idx continues after the first step's pair.
        assert seen_idx == [0, 1]
