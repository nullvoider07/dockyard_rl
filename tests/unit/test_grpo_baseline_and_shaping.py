"""F0: coverage for the machinery Track F builds on.

`calculate_baseline_and_std_per_prompt`, `GRPOAdvantageEstimator`,
`apply_reward_shaping` and `dynamic_sampling` are the four pieces every later F
batch rests on, and none of them had a single unit test. Track F prices a cost
column against a group pass rate, gates on group variance, and needs a
correctness column that survives shaping -- all three questions are answered by
code that was, until now, only exercised incidentally.

This batch changes no source. It records what these functions actually do,
including four behaviours that are surprising enough to plan around:

  * `prompts` must be 2-D. The docstring says `(batch, seq) or (batch,)`; the
    1-D form raises, because grouping is `(prompts == unique[i]).all(1)`.
  * A group of 2 under leave-one-out has `std == 0` always. The Bessel factor is
    `num_valid/(num_valid-1) = 1/0`, and the variance it multiplies is exactly 0,
    so `0 * inf -> nan -> nan_to_num(0)`. With `use_dynamic_sampling` on, every
    prompt is then discarded and the run dies at the gen-batch limit.
  * `valid_mask` excludes a sample from its group's baseline but does not zero
    that sample's own advantage -- it still receives the group baseline.
  * `apply_reward_shaping` writes `unshaped_total_reward` *after* its
    `enabled` early return, so with shaping off the column does not exist. Track F
    must carry its own correctness column rather than inherit this one.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from dockyard_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator
from dockyard_rl.algorithms.reward_functions import apply_reward_shaping
from dockyard_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict


def _shape(batch: Any, cfg: dict[str, Any]) -> Any:
    """`apply_reward_shaping` with the TypedDict cast kept out of every test."""
    return apply_reward_shaping(batch, cast(Any, cfg))


def _group(n: int, prompt_id: int = 7) -> torch.Tensor:
    """`prompts` as the trainer supplies it: a 2-D padded token-id matrix."""
    return torch.full((n, 1), prompt_id, dtype=torch.long)


def _baseline(prompts, rewards, *, loo=True, valid=None, std_rewards=None):
    valid = torch.ones_like(rewards) if valid is None else valid
    return calculate_baseline_and_std_per_prompt(
        prompts, rewards, valid, leave_one_out_baseline=loo, std_rewards=std_rewards
    )


class TestBaselineShapeContract:
    def test_one_dimensional_prompts_raise_despite_the_docstring(self):
        """The docstring offers `(batch,)`; grouping's `.all(1)` refuses it."""
        with pytest.raises(IndexError):
            _baseline(torch.tensor([0, 0, 1, 1]), torch.tensor([1.0, 0.0, 1.0, 1.0]))

    def test_two_dimensional_prompts_group_by_exact_row_equality(self):
        prompts = torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2]])
        rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])

        baseline, std = _baseline(prompts, rewards, loo=False)

        torch.testing.assert_close(baseline, torch.tensor([0.5, 0.5, 1.0, 1.0]))
        assert std[0] > 0 and std[2] == 0

    def test_padding_differences_split_a_group(self):
        """Grouping is byte equality, so two paddings of one prompt are two groups."""
        prompts = torch.tensor([[5, 0], [5, 9]])
        rewards = torch.tensor([1.0, 0.0])

        baseline, _ = _baseline(prompts, rewards, loo=False)

        # Each row is alone in its group, so each baseline is its own reward.
        torch.testing.assert_close(baseline, rewards)


class TestBaselineValues:
    def test_leave_one_out_excludes_the_sample_itself(self):
        rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])

        baseline, _ = _baseline(_group(4), rewards, loo=True)

        # Each row sees the mean of the other three.
        torch.testing.assert_close(
            baseline, torch.tensor([1 / 3, 1 / 3, 2 / 3, 2 / 3])
        )

    def test_plain_mean_includes_it(self):
        rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])

        baseline, _ = _baseline(_group(4), rewards, loo=False)

        torch.testing.assert_close(baseline, torch.full((4,), 0.5))

    def test_a_constant_group_has_zero_std_and_zero_advantage(self):
        rewards = torch.ones(4)

        baseline, std = _baseline(_group(4), rewards, loo=False)

        torch.testing.assert_close(baseline, rewards)
        assert torch.all(std == 0)

    def test_a_group_of_one_zeroes_its_own_gradient(self):
        """baseline is set to the reward, so the advantage is exactly 0."""
        baseline, std = _baseline(_group(1), torch.tensor([0.7]))

        torch.testing.assert_close(baseline, torch.tensor([0.7]))
        assert std.item() == 0.0


class TestTheGroupOfTwoTrap:
    """`num_generations_per_prompt: 2` plus leave-one-out has no variance at all."""

    def test_std_is_always_zero_for_a_group_of_two_under_leave_one_out(self):
        for rewards in (
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, -3.0]),
            torch.tensor([0.25, 0.75]),
        ):
            _, std = _baseline(_group(2), rewards, loo=True)
            assert torch.all(std == 0), f"expected 0 std for {rewards.tolist()}"

    def test_the_same_group_has_variance_without_leave_one_out(self):
        """Pins that it is the Bessel factor, not the data."""
        _, std = _baseline(_group(2), torch.tensor([1.0, 0.0]), loo=False)

        assert std[0] > 0

    def test_the_baseline_is_still_meaningful(self):
        """Only the std degenerates; each row still gets the other's reward."""
        baseline, _ = _baseline(_group(2), torch.tensor([1.0, 0.0]), loo=True)

        torch.testing.assert_close(baseline, torch.tensor([0.0, 1.0]))


class TestValidMask:
    def test_a_masked_sample_is_excluded_from_the_baseline(self):
        rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
        valid = torch.tensor([1.0, 1.0, 1.0, 0.0])

        baseline, _ = _baseline(_group(4), rewards, loo=False, valid=valid)

        # (1 + 1 + 0) / 3
        torch.testing.assert_close(baseline, torch.full((4,), 2 / 3))

    def test_but_the_masked_sample_still_receives_a_baseline(self):
        """Masking does not zero its advantage -- the caller must do that."""
        rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
        valid = torch.tensor([1.0, 1.0, 1.0, 0.0])

        baseline, _ = _baseline(_group(4), rewards, loo=False, valid=valid)

        assert baseline[3] != 0.0
        assert (rewards[3] - baseline[3]).item() != 0.0

    def test_one_valid_sample_zeroes_the_group(self):
        valid = torch.tensor([1.0, 0.0, 0.0, 0.0])
        rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])

        baseline, _ = _baseline(_group(4), rewards, loo=False, valid=valid)

        torch.testing.assert_close(baseline, rewards)


class TestStdRewardsSeparation:
    """DAPO's hook: variance on the raw metric, baseline on the shaped reward."""

    def test_a_separate_std_tensor_changes_only_the_std(self):
        rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
        raw = torch.tensor([1.0, 0.0, 0.0, 0.0])

        baseline, std = _baseline(_group(4), rewards, loo=False, std_rewards=raw)
        baseline_plain, std_plain = _baseline(_group(4), rewards, loo=False)

        torch.testing.assert_close(baseline, baseline_plain)
        assert not torch.allclose(std, std_plain)

    def test_shaping_noise_cannot_rescue_a_degenerate_group(self):
        """The property Track F depends on: a saturated group stays at std 0.

        Every rollout solved the prompt, so the raw metric is constant. A shaped
        reward that varies (by cost, by length) must not make the group look
        informative.
        """
        shaped = torch.tensor([0.9, 0.7, 0.5, 0.3])
        raw = torch.ones(4)

        _, std = _baseline(_group(4), shaped, loo=False, std_rewards=raw)

        assert torch.all(std == 0)

    def test_without_the_hook_the_shaped_spread_does_look_informative(self):
        """Pins why the hook exists."""
        shaped = torch.tensor([0.9, 0.7, 0.5, 0.3])

        _, std = _baseline(_group(4), shaped, loo=False)

        assert torch.all(std > 0)

    def test_an_equal_valued_copy_gives_the_same_answer(self):
        """The `is` check is a shortcut, not a semantic switch."""
        rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])

        _, std_identity = _baseline(_group(4), rewards, loo=False)
        _, std_copy = _baseline(
            _group(4), rewards, loo=False, std_rewards=rewards.clone()
        )

        torch.testing.assert_close(std_identity, std_copy)


class TestGRPOAdvantageEstimator:
    def _estimator(self, *, normalize=True, loo=True):
        return GRPOAdvantageEstimator(
            cast(Any, {"normalize_rewards": normalize, "use_leave_one_out_baseline": loo}),
            cast(Any, {}),  # loss_config is unused by this estimator
        )

    def test_advantage_is_reward_minus_group_baseline(self):
        rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
        mask = torch.ones(4, 3)

        adv = self._estimator(normalize=False, loo=False).compute_advantage(
            prompt_ids=_group(4), rewards=rewards, mask=mask
        )

        torch.testing.assert_close(adv[:, 0], torch.tensor([0.5, 0.5, -0.5, -0.5]))

    def test_every_response_token_gets_the_same_scalar(self):
        adv = self._estimator(normalize=False, loo=False).compute_advantage(
            prompt_ids=_group(4),
            rewards=torch.tensor([1.0, 1.0, 0.0, 0.0]),
            mask=torch.ones(4, 5),
        )

        assert adv.shape == (4, 5)
        for row in adv:
            assert torch.all(row == row[0])

    def test_the_mask_is_expanded_to_not_multiplied_in(self):
        """A zero in the mask does not zero the advantage here."""
        mask = torch.tensor([[1.0, 0.0, 0.0]] * 4)

        adv = self._estimator(normalize=False, loo=False).compute_advantage(
            prompt_ids=_group(4),
            rewards=torch.tensor([1.0, 1.0, 0.0, 0.0]),
            mask=mask,
        )

        assert adv[0, 1] != 0.0

    def test_normalization_divides_by_the_group_std(self):
        rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
        _, std = _baseline(_group(4), rewards, loo=False)

        plain = self._estimator(normalize=False, loo=False).compute_advantage(
            prompt_ids=_group(4), rewards=rewards, mask=torch.ones(4, 1)
        )
        normed = self._estimator(normalize=True, loo=False).compute_advantage(
            prompt_ids=_group(4), rewards=rewards, mask=torch.ones(4, 1)
        )

        torch.testing.assert_close(
            normed[:, 0], plain[:, 0] / (std + 1e-6), rtol=1e-4, atol=1e-6
        )

    def test_a_zero_std_group_is_left_undivided_rather_than_exploding(self):
        adv = self._estimator(normalize=True, loo=False).compute_advantage(
            prompt_ids=_group(4), rewards=torch.ones(4), mask=torch.ones(4, 1)
        )

        assert torch.all(adv == 0)
        assert torch.isfinite(adv).all()

    def test_the_config_keys_are_required(self):
        """Plain indexing, not .get -- an incomplete estimator config fails loudly."""
        with pytest.raises(KeyError):
            GRPOAdvantageEstimator(cast(Any, {"normalize_rewards": True}), cast(Any, {}))

    def test_kwargs_are_ignored_so_a_new_column_can_ride(self):
        """Track F's cost column reaches the estimator through **kwargs."""
        adv = self._estimator(normalize=False, loo=False).compute_advantage(
            prompt_ids=_group(4),
            rewards=torch.tensor([1.0, 1.0, 0.0, 0.0]),
            mask=torch.ones(4, 1),
            some_future_column=torch.rand(4),
        )

        torch.testing.assert_close(adv[:, 0], torch.tensor([0.5, 0.5, -0.5, -0.5]))


def _shaping_batch(rewards, *, truncated=None, response_lengths=None):
    batch: Any = BatchedDataDict({"total_reward": rewards})
    if truncated is not None:
        batch["truncated"] = truncated
    if response_lengths is not None:
        batch["message_log"] = [
            [{"role": "assistant", "token_ids": torch.zeros(n, dtype=torch.long)}]
            for n in response_lengths
        ]
    return batch


class TestRewardShapingDisabled:
    def test_disabled_leaves_the_reward_alone(self):
        rewards = torch.tensor([1.0, 0.5])
        batch = _shaping_batch(rewards.clone())

        _shape(batch, {"enabled": False})

        torch.testing.assert_close(batch["total_reward"], rewards)

    def test_disabled_does_not_write_the_unshaped_column(self):
        """The correction Track F depends on: the snapshot is past the early return.

        A design that reads `unshaped_total_reward` as a correctness column gets
        a KeyError on any run with shaping off, which is every shipped config.
        """
        batch = _shaping_batch(torch.tensor([1.0, 0.5]))

        _shape(batch, {"enabled": False})

        assert "unshaped_total_reward" not in batch


class TestStopProperlyPenalty:
    def test_truncated_rewards_are_scaled_and_others_are_not(self):
        batch = _shaping_batch(
            torch.tensor([1.0, 1.0]), truncated=torch.tensor([True, False])
        )

        _shape(
            batch, {"enabled": True, "stop_properly_penalty_coef": 0.25}
        )

        torch.testing.assert_close(batch["total_reward"], torch.tensor([0.25, 1.0]))

    def test_the_unshaped_column_is_written_when_enabled(self):
        batch = _shaping_batch(
            torch.tensor([1.0, 1.0]), truncated=torch.tensor([True, False])
        )

        _shape(
            batch, {"enabled": True, "stop_properly_penalty_coef": 0.0}
        )

        torch.testing.assert_close(
            batch["unshaped_total_reward"], torch.tensor([1.0, 1.0])
        )

    def test_no_truncated_samples_leaves_the_reward_untouched(self):
        rewards = torch.tensor([1.0, 0.5])
        batch = _shaping_batch(rewards.clone(), truncated=torch.tensor([False, False]))

        _shape(
            batch, {"enabled": True, "stop_properly_penalty_coef": 0.0}
        )

        torch.testing.assert_close(batch["total_reward"], rewards)

    def test_a_missing_truncated_column_is_refused(self):
        batch = _shaping_batch(torch.tensor([1.0]))

        with pytest.raises(AssertionError, match="truncated"):
            apply_reward_shaping(
                batch, {"enabled": True, "stop_properly_penalty_coef": 0.5}
            )

    def test_a_coefficient_outside_the_unit_interval_is_refused(self):
        batch = _shaping_batch(torch.tensor([1.0]), truncated=torch.tensor([True]))

        with pytest.raises(AssertionError):
            apply_reward_shaping(
                batch, {"enabled": True, "stop_properly_penalty_coef": 1.5}
            )

    def test_it_short_circuits_the_dapo_branch(self):
        """The two modes are mutually exclusive; DAPO params are ignored here."""
        batch = _shaping_batch(
            torch.tensor([1.0]),
            truncated=torch.tensor([False]),
            response_lengths=[10_000],
        )

        apply_reward_shaping(
            batch,
            {
                "enabled": True,
                "stop_properly_penalty_coef": 1.0,
                "overlong_buffer_length": 10,
                "overlong_buffer_penalty": 1.0,
                "max_response_length": 20,
            },
        )

        torch.testing.assert_close(batch["total_reward"], torch.tensor([1.0]))


class TestDapoOverlongPenalty:
    CFG = {
        "enabled": True,
        "stop_properly_penalty_coef": None,
        "overlong_buffer_length": 10,
        "overlong_buffer_penalty": 1.0,
        "max_response_length": 30,
    }

    def test_a_response_within_budget_is_unpenalized(self):
        # expected_response_length = 30 - 10 = 20
        batch = _shaping_batch(torch.tensor([1.0]), response_lengths=[15])

        _shape(batch, dict(self.CFG))

        torch.testing.assert_close(batch["total_reward"], torch.tensor([1.0]))

    def test_the_penalty_ramps_across_the_buffer(self):
        batch = _shaping_batch(torch.tensor([1.0, 1.0, 1.0]), response_lengths=[20, 25, 30])

        _shape(batch, dict(self.CFG))

        # exceed = 0, 5, 10 over a 10-token buffer at penalty 1.0
        torch.testing.assert_close(
            batch["total_reward"], torch.tensor([1.0, 0.5, 0.0])
        )

    def test_missing_parameters_are_refused_with_a_reason(self):
        batch = _shaping_batch(torch.tensor([1.0]), response_lengths=[15])
        cfg = dict(self.CFG)
        cfg["overlong_buffer_penalty"] = None

        with pytest.raises(ValueError, match="overlong_buffer_penalty"):
            _shape(batch, cfg)

    def test_only_the_first_assistant_message_is_measured(self):
        """Multi-turn rollouts are scored on turn one, not the total."""
        batch: Any = BatchedDataDict({"total_reward": torch.tensor([1.0])})
        batch["message_log"] = [
            [
                {"role": "assistant", "token_ids": torch.zeros(20, dtype=torch.long)},
                {"role": "assistant", "token_ids": torch.zeros(999, dtype=torch.long)},
            ]
        ]

        _shape(batch, dict(self.CFG))

        torch.testing.assert_close(batch["total_reward"], torch.tensor([1.0]))


class _Timer:
    @contextlib.contextmanager
    def time(self, _name):
        yield


def _master_config(**grpo):
    base = {
        "num_prompts_per_step": 2,
        "num_generations_per_prompt": 2,
        "use_dynamic_sampling": True,
        "dynamic_sampling_max_gen_batches": 10,
    }
    base.update(grpo)
    return SimpleNamespace(grpo=base)


def _sampling_batch(n: int) -> BatchedDataDict:
    return BatchedDataDict(
        {
            "total_reward": torch.arange(n, dtype=torch.float32),
            "idx": torch.arange(n),
        }
    )


class TestDynamicSampling:
    def _run(self, batch, std, baseline, cfg, cache=None):
        from dockyard_rl.algorithms.grpo import dynamic_sampling

        return dynamic_sampling(
            batch, std, baseline, 0, cast(Any, cfg), _Timer(), batch_cache=cache
        )

    def test_disabled_returns_the_batch_untouched_but_stamps_the_statistics(self):
        batch = _sampling_batch(4)
        std = torch.zeros(4)
        baseline = torch.ones(4)

        out, complete, _cache, _metrics = self._run(
            batch, std, baseline, _master_config(use_dynamic_sampling=False)
        )

        assert complete is True
        assert out is batch
        torch.testing.assert_close(out["std"], std)
        torch.testing.assert_close(out["baseline"], baseline)

    def test_zero_variance_rows_are_dropped(self):
        batch = _sampling_batch(4)
        std = torch.tensor([1.0, 1.0, 0.0, 0.0])

        out, complete, _cache, _metrics = self._run(
            batch, std, torch.zeros(4), _master_config(num_prompts_per_step=1)
        )

        assert complete is True
        torch.testing.assert_close(out["filtered_reward"], torch.tensor([0.0, 1.0]))

    def test_a_fully_degenerate_batch_never_completes(self):
        """The group-of-2 trap surfaces here: no variance anywhere, no batch."""
        batch = _sampling_batch(4)

        out, complete, cache, _metrics = self._run(
            batch, torch.zeros(4), torch.zeros(4), _master_config()
        )

        assert complete is False
        assert cache is None or cache.size == 0

    def test_the_gen_batch_limit_is_enforced(self):
        from dockyard_rl.algorithms.grpo import dynamic_sampling

        with pytest.raises(ValueError, match="maximum allowed number of batches"):
            dynamic_sampling(
                _sampling_batch(4),
                torch.zeros(4),
                torch.zeros(4),
                11,
                cast(Any, _master_config(dynamic_sampling_max_gen_batches=10)),
                _Timer(),
            )

    def test_every_column_agrees_with_the_filtered_batch_size(self):
        """The invariant the old code broke by writing a [B_orig] column back."""
        batch = _sampling_batch(4)
        std = torch.tensor([1.0, 1.0, 0.0, 0.0])

        out, _complete, _cache, _metrics = self._run(
            batch, std, torch.zeros(4), _master_config(num_prompts_per_step=1)
        )

        assert out.size == 2
        for key in ("total_reward", "filtered_reward", "std", "baseline", "idx"):
            assert out[key].shape[0] == 2, f"{key} disagrees with the batch size"

    def test_the_unfiltered_rewards_ride_the_metrics_dict(self):
        """They cannot ride the batch: a [B_orig] column on a [B_kept] batch raises.

        The logged reward curve still needs every rollout, not just the ones
        that cleared the variance gate, so the tensor moves rather than dies.
        """
        batch = _sampling_batch(4)
        std = torch.tensor([1.0, 1.0, 0.0, 0.0])

        _out, _complete, _cache, metrics = self._run(
            batch, std, torch.zeros(4), _master_config(num_prompts_per_step=1)
        )

        torch.testing.assert_close(
            metrics["dynamic_sampling_unfiltered_reward"],
            torch.arange(4, dtype=torch.float32),
        )

    def test_the_caller_renames_it_onto_the_reward_metric(self):
        """Cut the rename and the raw key ships while `reward` goes missing."""
        import inspect

        from dockyard_rl.algorithms import grpo

        source = inspect.getsource(grpo.grpo_train)
        assert "metrics.pop(" in source
        assert '"dynamic_sampling_unfiltered_reward"' in source


class TestBatchedDataDictHasNoSliceMethod:
    """Two call sites invoked `.slice(a, b)`, which does not exist on this type.

    `dynamic_sampling` (`grpo.py`) and `SequencePackingLossWrapper.__call__`
    (`algorithms/loss/wrapper.py`) both did. The second is the serious one: it
    runs on every packed micro-batch, and `sequence_packing.enabled: true` in
    `grpo_swe.yaml` and eleven other shipped configs. It was unreachable only
    because packing itself raised `KeyError` on an unregistered algorithm name
    until that was fixed in C3 -- so fixing C3 unmasked this, exactly as C3's
    own note predicted for the defect it unmasked.
    """

    def test_the_method_really_is_absent(self):
        assert not hasattr(BatchedDataDict({"a": torch.zeros(2)}), "slice")

    def test_select_takes_a_slice_object_instead(self):
        batch = BatchedDataDict({"a": torch.arange(5), "b": torch.arange(5) * 2})

        out = batch.select(slice(1, 3))

        torch.testing.assert_close(out["a"], torch.tensor([1, 2]))
        torch.testing.assert_close(out["b"], torch.tensor([2, 4]))

    @pytest.mark.parametrize(
        "relpath", ["algorithms/grpo.py", "algorithms/loss/wrapper.py"]
    )
    def test_neither_call_site_came_back(self, relpath):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        source = (root / relpath).read_text()
        code = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )
        assert ".slice(" not in code, (
            f"{relpath} calls .slice() again; BatchedDataDict has no such method "
            "and this raises AttributeError at runtime. Use .select(slice(a, b))."
        )
