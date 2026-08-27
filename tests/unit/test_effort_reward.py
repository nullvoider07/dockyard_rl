"""F2: the difficulty-gated effort reward, and the bound that makes it safe.

The term prices generated tokens against the group's own pass rate: full
pressure where the group already solves the prompt, effectively none where it
does not. No difficulty labels exist anywhere -- the per-prompt baseline supplies
the difficulty read.

Two things get more scrutiny here than the wiring does.

**The bound.** An earlier revision of the plan asserted that an unwired `p̄` was
itself the catastrophic failure -- that the reward would invert and train the
model to give up on hard problems. That was asserted, never derived, and it is
wrong. A success outranks a failure exactly when
`λ·(cost_success − cost_failure) < 1`, and since cost is clamped to [0, 1],
`λ < 1` is sufficient on its own. `p̄` makes the term *useful*; the clamp and the
bound make it *safe*. `TestTheBoundIsWhatPreventsInversion` is a property test
over randomized groups rather than a worked example, because a worked example is
what let the wrong claim stand for four revisions.

**The refusals.** Three of the four advantage estimators cannot consume a shaped
`total_reward` at all -- gdpo reads only its `reward/<name>` components, opd
derives its advantage from teacher logprobs and touches neither rewards nor
prompt_ids, reinforce_plus_plus discards the group std and normalizes globally.
Enabled under any of them the term computes a reward that is then discarded:
a config that reads as live and does nothing, which is the exact defect class
Track F carries the most risk of.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, cast

import pytest
import torch

from dockyard_rl.algorithms.effort_reward import (
    SUPPORTED_ESTIMATORS,
    apply_effort_reward,
    effective_coefficient,
    effort_reward_metrics,
    group_pass_rate,
    normalize_effort_cost,
    response_token_counts,
    validate_effort_reward_config,
)
from dockyard_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

CFG: dict[str, Any] = {
    "enabled": True,
    "coefficient": 0.8,
    "difficulty_exponent": 2.0,
    "difficulty_floor": 0.0,
    "max_response_tokens": 100,
}


def _log(*token_counts_per_turn: int, role: str = "assistant") -> list[dict]:
    return [
        {"role": role, "token_ids": torch.zeros(n, dtype=torch.long)}
        for n in token_counts_per_turn
    ]


def _batch(rewards, logs, *, prompt_ids=None):
    batch: Any = BatchedDataDict({"total_reward": torch.tensor(rewards)})
    batch["message_log"] = logs
    return batch, (
        torch.full((len(rewards), 1), 7, dtype=torch.long)
        if prompt_ids is None
        else prompt_ids
    )


class TestEffortCost:
    def test_every_assistant_turn_counts_not_only_the_first(self):
        """DAPO's overlong penalty reads the first turn; effort is the whole trajectory."""
        counts = response_token_counts([_log(10, 20, 30)])

        assert counts.tolist() == [60.0]

    def test_non_assistant_turns_are_excluded(self):
        log = _log(10) + [
            {"role": "user", "token_ids": torch.zeros(500, dtype=torch.long)}
        ]

        assert response_token_counts([log]).tolist() == [10.0]

    def test_a_turn_without_token_ids_is_skipped(self):
        log = _log(10) + [{"role": "assistant"}]

        assert response_token_counts([log]).tolist() == [10.0]

    def test_normalization_maps_onto_the_unit_interval(self):
        cost = normalize_effort_cost(torch.tensor([0.0, 50.0, 100.0]), 100)

        torch.testing.assert_close(cost, torch.tensor([0.0, 0.5, 1.0]))

    def test_an_overlong_rollout_clamps_rather_than_exceeding_one(self):
        """The clamp is the safety property: the bound assumes cost <= 1."""
        cost = normalize_effort_cost(torch.tensor([10_000.0]), 100)

        assert cost.item() == 1.0

    def test_a_non_positive_normalizer_is_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            normalize_effort_cost(torch.tensor([1.0]), 0)


class TestDifficultyScaling:
    def test_pass_rate_is_the_group_mean_of_correctness(self):
        prompts = torch.tensor([[1], [1], [1], [1], [2], [2]])
        correctness = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])

        rate = group_pass_rate(prompts, correctness)

        torch.testing.assert_close(
            rate, torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0, 1.0])
        )

    def test_it_is_not_the_baseline_the_trainer_already_computes(self):
        """`p̄` is the mean of correctness; the existing call means the reward.

        They coincide only before shaping. This term replaces the reward, so
        reusing the existing baseline would feed `λ_eff` its own output.
        """
        prompts = torch.tensor([[1]] * 4)
        correctness = torch.tensor([1.0, 1.0, 0.0, 0.0])
        shaped = torch.tensor([0.9, 0.7, 0.0, 0.0])

        rate = group_pass_rate(prompts, correctness)
        reward_baseline, _ = calculate_baseline_and_std_per_prompt(
            prompts, shaped, torch.ones(4), leave_one_out_baseline=False
        )

        assert not torch.allclose(rate, reward_baseline)

    def test_the_tax_is_effectively_off_on_a_hard_prompt(self):
        lam_eff = effective_coefficient(torch.tensor([0.25]), CFG)

        assert lam_eff.item() == pytest.approx(0.8 * 0.25**2)
        assert lam_eff.item() < 0.06

    def test_the_tax_is_full_strength_on_a_saturated_prompt(self):
        lam_eff = effective_coefficient(torch.tensor([1.0]), CFG)

        assert lam_eff.item() == pytest.approx(0.8)

    def test_a_linear_exponent_reproduces_alps_scaling(self):
        cfg = dict(CFG, difficulty_exponent=1.0)

        lam_eff = effective_coefficient(torch.tensor([0.25]), cfg)

        assert lam_eff.item() == pytest.approx(0.8 * 0.25)

    def test_the_floor_keeps_a_minimum_penalty_on_a_zero_scoring_group(self):
        """ALP's `max(acc, 1/G)`; without it a 0/8 group is taxed at exactly 0."""
        floored = effective_coefficient(torch.tensor([0.0]), dict(CFG, difficulty_floor=0.125))
        unfloored = effective_coefficient(torch.tensor([0.0]), CFG)

        assert unfloored.item() == 0.0
        assert floored.item() == pytest.approx(0.8 * 0.125**2)

    def test_the_exponent_is_the_whole_difference_from_alp(self):
        """Pins the 8x gap the plan's original gamma=2 hid."""
        linear = effective_coefficient(torch.tensor([0.125]), dict(CFG, difficulty_exponent=1.0))
        square = effective_coefficient(torch.tensor([0.125]), CFG)

        assert linear.item() / square.item() == pytest.approx(8.0)


class TestApplyingTheTerm:
    def test_a_failed_rollout_is_pinned_at_zero_whatever_it_cost(self):
        """Cost only ever ranks within the successful set."""
        batch, prompts = _batch([0.0, 0.0], [_log(5), _log(100)])

        apply_effort_reward(batch, CFG, prompt_ids=prompts)

        torch.testing.assert_close(batch["total_reward"], torch.tensor([0.0, 0.0]))

    def test_failures_stay_identical_even_when_the_tax_is_live(self):
        """The multiplicative form, pinned where an additive tax would differ.

        The previous test has p-bar = 0, so lambda_eff is 0 and both forms agree.
        Here one rollout succeeds, so the tax is on -- and a subtracted tax would
        push the expensive failure below the cheap one, which is the ranking that
        teaches a model to give up early.
        """
        batch, prompts = _batch(
            [1.0, 0.0, 0.0, 0.0], [_log(50), _log(100), _log(0), _log(50)]
        )

        apply_effort_reward(batch, CFG, prompt_ids=prompts)

        failed = batch["total_reward"][1:]
        assert torch.all(failed == failed[0]), (
            "failed rollouts must be indistinguishable regardless of cost; "
            "a subtracted tax ranks the cheap failure above the expensive one"
        )

    def test_the_reward_is_never_negative(self):
        """A subtracted tax makes failures negative; this form floors them at 0."""
        batch, prompts = _batch(
            [1.0, 0.0, 0.0, 0.0], [_log(100), _log(100), _log(100), _log(100)]
        )

        apply_effort_reward(batch, CFG, prompt_ids=prompts)

        assert torch.all(batch["total_reward"] >= 0.0)

    def test_a_cheap_success_outranks_an_expensive_one(self):
        batch, prompts = _batch([1.0, 1.0], [_log(10), _log(90)])

        apply_effort_reward(batch, CFG, prompt_ids=prompts)

        assert batch["total_reward"][0] > batch["total_reward"][1]

    def test_the_worked_example_from_the_plan(self):
        """hard 2/8 at cost 0.9 keeps 95.5% of full reward."""
        rewards = [1.0, 1.0] + [0.0] * 6
        logs = [_log(90), _log(90)] + [_log(10)] * 6
        batch, prompts = _batch(rewards, logs)

        apply_effort_reward(batch, CFG, prompt_ids=prompts)

        assert batch["total_reward"][0].item() == pytest.approx(0.955, abs=1e-3)

    def test_disabled_is_a_no_op(self):
        batch, prompts = _batch([1.0, 0.0], [_log(90), _log(10)])
        before = batch["total_reward"].clone()

        apply_effort_reward(batch, dict(CFG, enabled=False), prompt_ids=prompts)

        torch.testing.assert_close(batch["total_reward"], before)
        assert "effort_cost" not in batch

    def test_a_missing_config_is_a_no_op(self):
        batch, prompts = _batch([1.0], [_log(90)])

        apply_effort_reward(batch, None, prompt_ids=prompts)

        torch.testing.assert_close(batch["total_reward"], torch.tensor([1.0]))

    def test_it_writes_its_own_correctness_column(self):
        """`unshaped_total_reward` is absent unless shaping ran; do not inherit it."""
        batch, prompts = _batch([1.0, 0.0], [_log(50), _log(50)])

        apply_effort_reward(batch, CFG, prompt_ids=prompts)

        torch.testing.assert_close(
            batch["unshaped_total_reward"], torch.tensor([1.0, 0.0])
        )

    def test_it_does_not_overwrite_an_existing_correctness_column(self):
        """Shaping ran first; its snapshot is the task metric, not this term's input."""
        batch, prompts = _batch([0.4, 0.0], [_log(50), _log(50)])
        batch["unshaped_total_reward"] = torch.tensor([1.0, 0.0])

        apply_effort_reward(batch, CFG, prompt_ids=prompts)

        torch.testing.assert_close(
            batch["unshaped_total_reward"], torch.tensor([1.0, 0.0])
        )

    def test_the_cost_column_is_published(self):
        batch, prompts = _batch([1.0, 1.0], [_log(25), _log(75)])

        apply_effort_reward(batch, CFG, prompt_ids=prompts)

        torch.testing.assert_close(batch["effort_cost"], torch.tensor([0.25, 0.75]))

    def test_correctness_outside_the_unit_interval_is_refused(self):
        """MathMultiRewardEnvironment sums three binaries and reaches 3.0."""
        batch, prompts = _batch([3.0, 1.0], [_log(10), _log(10)])

        with pytest.raises(ValueError, match=r"correctness in \[0, 1\]"):
            apply_effort_reward(batch, CFG, prompt_ids=prompts)

    def test_the_refusal_names_the_environment_that_causes_it(self):
        batch, prompts = _batch([-1.0], [_log(10)])

        with pytest.raises(ValueError, match="MultiReward"):
            apply_effort_reward(batch, CFG, prompt_ids=prompts)


class TestTheBoundIsWhatPreventsInversion:
    """Property test, not a worked example.

    The claim that `p̄` was the safety wire survived four plan revisions on the
    strength of a worked example. This randomizes instead.
    """

    @staticmethod
    def _advantages(correctness, cost, lam_eff, *, loo, normalize):
        rewards = correctness * (1.0 - lam_eff * cost)
        n = len(rewards)
        total = rewards.sum()
        if loo:
            adv = (n * rewards - total) / (n - 1)
        else:
            adv = rewards - total / n
        if normalize:
            adv = adv / (rewards.std(unbiased=False) + 1e-8)
        return adv

    @pytest.mark.parametrize(
        "loo,normalize", list(itertools.product([False, True], [False, True]))
    )
    def test_no_group_inverts_under_the_bound(self, loo, normalize):
        rng = random.Random(20260827)
        checked = 0
        for _ in range(4000):
            n = rng.choice([4, 8, 16])
            correctness = torch.tensor(
                [1.0 if rng.random() < rng.choice([0.125, 0.25, 0.5, 0.875]) else 0.0
                 for _ in range(n)]
            )
            if correctness.sum() in (0, n):
                continue  # a group with no spread contributes no gradient anyway
            cost = torch.rand(n)
            lam_eff = rng.uniform(0.01, 0.99)  # the bound: lam * cost_max < 1

            adv = self._advantages(
                correctness, cost, lam_eff, loo=loo, normalize=normalize
            )
            succeeded = adv[correctness == 1.0]
            failed = adv[correctness == 0.0]
            assert succeeded.min() > failed.max(), (
                f"inverted at lam_eff={lam_eff:.3f}, loo={loo}, norm={normalize}"
            )
            checked += 1
        assert checked > 500, "the sampler produced too few usable groups"

    def test_it_does_invert_once_the_bound_is_broken(self):
        """Pins that the test can fail -- otherwise it proves nothing."""
        correctness = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cost = torch.tensor([0.9, 0.85, 0.1, 0.1, 0.12, 0.08, 0.1, 0.1])

        adv = self._advantages(correctness, cost, 2.0, loo=False, normalize=False)

        assert adv[correctness == 1.0].min() < adv[correctness == 0.0].max()

    def test_the_clamp_is_what_keeps_the_bound_true(self):
        """A cost above 1 satisfies `lam < 1` on paper and inverts in fact."""
        correctness = torch.tensor([1.0, 0.0])
        unclamped = torch.tensor([5.0, 0.0])

        adv = self._advantages(
            correctness, unclamped, 0.9, loo=False, normalize=False
        )

        assert adv[0] < adv[1]


class TestStartupRefusals:
    def _validate(self, cfg, *, estimator="grpo", trainer="sync"):
        validate_effort_reward_config(
            cast(Any, cfg), estimator_name=estimator, trainer=trainer
        )

    def test_a_valid_config_passes(self):
        self._validate(CFG)

    def test_disabled_is_never_validated(self):
        """Inert default: a nonsense disabled block must not block a run."""
        self._validate({"enabled": False, "coefficient": 99.0})

    def test_none_is_never_validated(self):
        self._validate(None)

    @pytest.mark.parametrize("coefficient", [1.0, 1.5, -0.1])
    def test_a_coefficient_outside_the_bound_is_refused(self, coefficient):
        with pytest.raises(ValueError, match=r"coefficient must be in \[0, 1\)"):
            self._validate(dict(CFG, coefficient=coefficient))

    def test_the_coefficient_refusal_explains_the_consequence(self):
        with pytest.raises(ValueError, match="give up early"):
            self._validate(dict(CFG, coefficient=2.0))

    def test_a_missing_normalizer_is_refused(self):
        with pytest.raises(ValueError, match="max_response_tokens"):
            self._validate(dict(CFG, max_response_tokens=0))

    def test_a_negative_exponent_is_refused(self):
        with pytest.raises(ValueError, match="difficulty_exponent"):
            self._validate(dict(CFG, difficulty_exponent=-1.0))

    def test_a_floor_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValueError, match="difficulty_floor"):
            self._validate(dict(CFG, difficulty_floor=1.5))

    @pytest.mark.parametrize("estimator", ["gdpo", "opd", "reinforce_plus_plus"])
    def test_an_estimator_that_cannot_consume_it_is_refused(self, estimator):
        with pytest.raises(ValueError, match="per-prompt group baseline"):
            self._validate(CFG, estimator=estimator)

    def test_the_estimator_refusal_says_why_each_one_cannot(self):
        with pytest.raises(ValueError) as exc:
            self._validate(CFG, estimator="opd")
        assert "teacher logprobs" in str(exc.value)

    def test_only_grpo_is_supported(self):
        assert SUPPORTED_ESTIMATORS == {"grpo"}

    @pytest.mark.parametrize("trainer", ["async", "dataplane"])
    def test_a_trainer_without_the_reward_pipeline_is_refused(self, trainer):
        with pytest.raises(ValueError, match="not supported on the"):
            self._validate(CFG, trainer=trainer)


class TestMetrics:
    def test_metrics_need_no_prompt_ids_of_their_own(self):
        """The trainer deletes `input_ids` long before metrics are assembled.

        Reaching for it there is an UnboundLocalError on every sync step,
        whether or not the term is enabled -- pyright caught exactly that in the
        first draft of this wiring.
        """
        import inspect

        assert list(
            inspect.signature(effort_reward_metrics).parameters
        ) == ["batch"]

    def test_nothing_is_reported_when_the_term_did_not_run(self):
        batch, prompts = _batch([1.0], [_log(10)])

        assert effort_reward_metrics(batch) == {}

    def test_cost_is_reported_stratified_by_difficulty(self):
        """The mean hides the failure: short hard rollouts, long easy ones."""
        prompts = torch.tensor([[1]] * 4 + [[2]] * 4)
        rewards = [1.0, 0.0, 0.0, 0.0] + [1.0] * 4
        logs = [_log(90)] * 4 + [_log(20)] * 4
        batch, _ = _batch(rewards, logs)

        apply_effort_reward(batch, CFG, prompt_ids=prompts)
        metrics = effort_reward_metrics(batch)

        assert metrics["effort/cost_mean_hard"] == pytest.approx(0.9)
        assert metrics["effort/cost_mean_easy"] == pytest.approx(0.2)


class TestItIsWiredIntoTheSyncPipeline:
    def test_the_trainer_applies_it(self):
        import inspect

        from dockyard_rl.algorithms import grpo

        source = inspect.getsource(grpo.grpo_train)
        assert "repeated_batch = apply_effort_reward(" in source, (
            "the sync trainer no longer applies the effort term to the batch it "
            "goes on to use; calling it and discarding the result prices nothing"
        )

    def test_it_runs_before_the_baseline_is_taken(self):
        import inspect

        from dockyard_rl.algorithms import grpo

        source = inspect.getsource(grpo.grpo_train)
        assert source.index("apply_effort_reward(") < source.index(
            "calculate_baseline_and_std_per_prompt("
        ), "the advantage must be taken over the shaped reward, not before it"

    def test_setup_validates_the_config(self):
        import inspect

        from dockyard_rl.algorithms import grpo

        source = inspect.getsource(grpo.setup)
        assert "_validate_effort_reward_compat(" in source

    def test_the_async_entrypoint_refuses_it(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        source = (root / "examples" / "run_grpo_swe.py").read_text()
        assert '"effort_reward"' in source, (
            "async GRPO no longer refuses the effort reward; it would read as "
            "configured and never be applied"
        )

    def test_the_shipped_config_ships_it_disabled(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        text = (root / "examples" / "configs" / "grpo_swe.yaml").read_text()
        block = text.split("effort_reward:")[1]
        assert block.split("enabled:")[1].strip().startswith("false")
