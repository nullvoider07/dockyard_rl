"""Graded rollout allocation (F3a).

The unit under test is not the apportionment arithmetic on its own -- it is the
seam inside `dynamic_sampling` where the surplus is spent. F2 shipped a term
whose every isolated test passed while the term discarded the reward pipeline it
was supposed to compose with, because every test built a batch where the two
were indistinguishable. So the allocation is exercised through
`dynamic_sampling` here, with the batch shape the trainer actually hands it, and
the isolated tests below that are the arithmetic only.
"""

from __future__ import annotations

import contextlib
import inspect
import random
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from dockyard_rl.algorithms.rollout_allocation import (
    GRADED_ALLOCATION_GROUP_KEY,
    allocate_rollout_budget,
    assign_group_ids,
    graded_allocation_enabled,
    graded_allocation_metrics,
    validate_graded_allocation_config,
)
from dockyard_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict


class _Timer:
    @contextlib.contextmanager
    def time(self, _name):
        yield


def _master_config(**grpo):
    base: dict[str, Any] = {
        "num_prompts_per_step": 2,
        "num_generations_per_prompt": 2,
        "use_dynamic_sampling": True,
        "dynamic_sampling_max_gen_batches": 10,
    }
    base.update(grpo)
    return SimpleNamespace(grpo=base)


def _alloc(**kw):
    cfg = {"enabled": True, "min_rollouts_per_prompt": 1, "variance_exponent": 1.0}
    cfg.update(kw)
    return cfg


def _group_ids(sizes: list[int]) -> torch.Tensor:
    out: list[int] = []
    for gid, n in enumerate(sizes):
        out.extend([gid] * n)
    return torch.tensor(out, dtype=torch.int64)


def _row_weights(sizes: list[int], per_group: list[float]) -> torch.Tensor:
    out: list[float] = []
    for n, w in zip(sizes, per_group):
        out.extend([w] * n)
    return torch.tensor(out, dtype=torch.float32)


def _counts(group_ids: torch.Tensor, kept: list[int]) -> dict[int, int]:
    ids = group_ids.tolist()
    counts: dict[int, int] = {}
    for r in kept:
        counts[int(ids[r])] = counts.get(int(ids[r]), 0) + 1
    return counts


# ── The allocation arithmetic


class TestBudgetIsExact:
    def test_budget_at_or_above_batch_keeps_everything(self):
        sizes = [3, 3]
        kept = allocate_rollout_budget(
            _group_ids(sizes), _row_weights(sizes, [1.0, 2.0]), 6
        )
        assert kept == list(range(6))

    def test_budget_above_batch_keeps_everything(self):
        sizes = [2, 2]
        kept = allocate_rollout_budget(
            _group_ids(sizes), _row_weights(sizes, [1.0, 2.0]), 99
        )
        assert kept == list(range(4))

    def test_zero_budget_keeps_nothing(self):
        sizes = [2, 2]
        assert (
            allocate_rollout_budget(
                _group_ids(sizes), _row_weights(sizes, [1.0, 2.0]), 0
            )
            == []
        )

    def test_kept_rows_are_ascending_and_unique(self):
        sizes = [4, 4, 4]
        kept = allocate_rollout_budget(
            _group_ids(sizes), _row_weights(sizes, [0.1, 0.9, 0.5]), 7
        )
        assert kept == sorted(set(kept))

    @pytest.mark.parametrize("seed", range(25))
    def test_randomized_allocation_spends_the_budget_exactly(self, seed):
        rng = random.Random(seed)
        sizes = [rng.randint(1, 6) for _ in range(rng.randint(1, 7))]
        weights = [rng.random() for _ in sizes]
        total = sum(sizes)
        budget = rng.randint(1, total)
        kept = allocate_rollout_budget(
            _group_ids(sizes),
            _row_weights(sizes, weights),
            budget,
            min_rollouts_per_prompt=rng.randint(0, 3),
            variance_exponent=rng.choice([0.0, 1.0, 2.0]),
        )
        assert len(kept) == budget

    @pytest.mark.parametrize("seed", range(25))
    def test_randomized_allocation_never_exceeds_a_group(self, seed):
        rng = random.Random(seed)
        sizes = [rng.randint(1, 6) for _ in range(rng.randint(1, 7))]
        weights = [rng.random() for _ in sizes]
        group_ids = _group_ids(sizes)
        budget = rng.randint(1, sum(sizes))
        kept = allocate_rollout_budget(
            group_ids, _row_weights(sizes, weights), budget
        )
        counts = _counts(group_ids, kept)
        for gid, n in enumerate(sizes):
            assert counts.get(gid, 0) <= n

    def test_all_zero_weights_still_spend_the_budget(self):
        # Post-gate this should not happen, but a short return here would be a
        # silent under-fill of the train batch rather than an error.
        sizes = [3, 3]
        kept = allocate_rollout_budget(
            _group_ids(sizes),
            _row_weights(sizes, [0.0, 0.0]),
            4,
            min_rollouts_per_prompt=0,
        )
        assert len(kept) == 4


class TestGradingDirection:
    def test_the_high_variance_group_gets_more_rows(self):
        sizes = [4, 4]
        group_ids = _group_ids(sizes)
        kept = allocate_rollout_budget(
            group_ids, _row_weights(sizes, [0.05, 0.95]), 4
        )
        counts = _counts(group_ids, kept)
        assert counts[1] > counts[0]

    def test_exponent_zero_is_a_uniform_split(self):
        sizes = [4, 4]
        group_ids = _group_ids(sizes)
        kept = allocate_rollout_budget(
            group_ids,
            _row_weights(sizes, [0.05, 0.95]),
            4,
            variance_exponent=0.0,
        )
        assert _counts(group_ids, kept) == {0: 2, 1: 2}

    def test_a_higher_exponent_concentrates_the_budget(self):
        sizes = [6, 6]
        group_ids = _group_ids(sizes)
        gentle = _counts(
            group_ids,
            allocate_rollout_budget(
                group_ids,
                _row_weights(sizes, [0.4, 0.6]),
                6,
                variance_exponent=1.0,
            ),
        )
        sharp = _counts(
            group_ids,
            allocate_rollout_budget(
                group_ids,
                _row_weights(sizes, [0.4, 0.6]),
                6,
                variance_exponent=8.0,
            ),
        )
        assert sharp[1] > gentle[1]

    def test_the_floor_keeps_every_surviving_group_represented(self):
        sizes = [4, 4, 4]
        group_ids = _group_ids(sizes)
        kept = allocate_rollout_budget(
            group_ids,
            _row_weights(sizes, [0.001, 0.001, 0.998]),
            6,
            min_rollouts_per_prompt=1,
        )
        counts = _counts(group_ids, kept)
        assert set(counts) == {0, 1, 2}
        assert all(v >= 1 for v in counts.values())

    def test_a_zero_floor_lets_a_group_be_starved(self):
        sizes = [4, 4]
        group_ids = _group_ids(sizes)
        kept = allocate_rollout_budget(
            group_ids,
            _row_weights(sizes, [0.0001, 0.9999]),
            4,
            min_rollouts_per_prompt=0,
        )
        assert _counts(group_ids, kept).get(0, 0) == 0

    def test_floors_that_exceed_the_budget_go_to_the_highest_variance_groups(self):
        # Three groups, floor 2 each = 6 rows wanted, budget 4.
        sizes = [3, 3, 3]
        group_ids = _group_ids(sizes)
        kept = allocate_rollout_budget(
            group_ids,
            _row_weights(sizes, [0.1, 0.9, 0.5]),
            4,
            min_rollouts_per_prompt=2,
        )
        counts = _counts(group_ids, kept)
        assert len(kept) == 4
        assert counts.get(1, 0) == 2
        assert counts.get(2, 0) == 2
        assert counts.get(0, 0) == 0


class TestWithinAGroupSelectionIsSignalAgnostic:
    """Selecting rows inside a group by their own reward would bias the group's
    advantage -- it is a non-random subsample correlated with the quantity being
    estimated. Rows are taken in batch order, which is what truncation did."""

    def test_rows_within_a_group_are_taken_in_order(self):
        sizes = [4]
        kept = allocate_rollout_budget(
            _group_ids(sizes), _row_weights(sizes, [1.0]), 2
        )
        assert kept == [0, 1]

    def test_per_row_weight_order_does_not_reorder_within_a_group(self):
        group_ids = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        ascending = torch.tensor([0.1, 0.2, 0.3, 0.4])
        descending = torch.tensor([0.4, 0.3, 0.2, 0.1])
        assert allocate_rollout_budget(group_ids, ascending, 2) == [0, 1]
        assert allocate_rollout_budget(group_ids, descending, 2) == [0, 1]

    def test_the_group_weight_is_the_mean_not_the_first_row(self):
        # Group 0 leads with the batch's single largest std and still averages
        # lower than group 1. Reading any one row instead of the mean sends the
        # budget the wrong way, and leave-one-out guarantees rows differ.
        group_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)
        weights = torch.tensor([0.9, 0.0, 0.0, 0.2, 0.5, 0.5])
        kept = allocate_rollout_budget(
            group_ids, weights, 2, min_rollouts_per_prompt=0
        )
        assert _counts(group_ids, kept) == {1: 2}

    def test_group_weight_is_the_mean_so_leave_one_out_std_is_handled(self):
        # Under LOO, std varies within a group. The group's weight is the mean,
        # so a group whose rows average higher wins regardless of row order.
        group_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        weights = torch.tensor([0.9, 0.9, 0.1, 0.1])
        kept = allocate_rollout_budget(
            group_ids, weights, 2, min_rollouts_per_prompt=0
        )
        assert _counts(group_ids, kept) == {0: 2}


# ── Group ids


class TestAssignGroupIds:
    def test_a_one_dimensional_prompt_tensor_is_refused(self):
        with pytest.raises(ValueError, match="2-D"):
            assign_group_ids(torch.tensor([1, 2, 3]), gen_batch_ordinal=0)

    def test_a_negative_ordinal_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            assign_group_ids(torch.zeros(2, 3, dtype=torch.long), gen_batch_ordinal=-1)

    def test_partitions_identically_to_the_baseline_helper(self):
        prompts = torch.tensor(
            [[1, 1], [1, 1], [2, 2], [2, 2], [1, 1], [3, 3]], dtype=torch.long
        )
        gids = assign_group_ids(prompts, gen_batch_ordinal=0)

        by_id: dict[int, list[int]] = {}
        for row, gid in enumerate(gids.tolist()):
            by_id.setdefault(int(gid), []).append(row)

        # The helper's own partition, built the way it builds it.
        unique = torch.unique(prompts, dim=0)
        helper_groups = {
            tuple(
                torch.arange(len(prompts))[
                    (prompts == unique[i]).all(1)
                ].tolist()
            )
            for i in range(len(unique))
        }
        assert {tuple(v) for v in by_id.values()} == helper_groups

    def test_leave_one_out_std_varies_within_a_group_which_is_why_it_is_meaned(self):
        # Not a property of assign_group_ids -- a property of the weight it is
        # paired with, and the reason allocate_rollout_budget reduces a group's
        # rows by mean rather than taking any single row's std.
        prompts = torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.long)
        rewards = torch.tensor([1.0, 0.0, 0.5])
        _, loo = calculate_baseline_and_std_per_prompt(
            prompts, rewards, torch.ones_like(rewards), leave_one_out_baseline=True
        )
        _, plain = calculate_baseline_and_std_per_prompt(
            prompts, rewards, torch.ones_like(rewards), leave_one_out_baseline=False
        )
        assert len({round(float(v), 6) for v in loo}) > 1
        assert len({round(float(v), 6) for v in plain}) == 1

    def test_the_ordinal_separates_identical_prompts_across_generation_batches(self):
        prompts = torch.tensor([[7, 7], [7, 7]], dtype=torch.long)
        first = assign_group_ids(prompts, gen_batch_ordinal=0)
        second = assign_group_ids(prompts, gen_batch_ordinal=1)
        assert set(first.tolist()).isdisjoint(set(second.tolist()))

    def test_ids_are_dense_within_one_batch(self):
        prompts = torch.tensor([[1, 1], [2, 2], [1, 1]], dtype=torch.long)
        gids = assign_group_ids(prompts, gen_batch_ordinal=0).tolist()
        assert sorted(set(gids)) == [0, 1]


# ── Configuration refusals


class TestValidation:
    def test_absent_config_is_inert(self):
        validate_graded_allocation_config(
            None, use_dynamic_sampling=False, trainer="async"
        )

    def test_disabled_config_is_inert(self):
        validate_graded_allocation_config(
            _alloc(enabled=False), use_dynamic_sampling=False, trainer="async"
        )

    def test_dynamic_sampling_off_is_refused(self):
        with pytest.raises(ValueError, match="use_dynamic_sampling"):
            validate_graded_allocation_config(
                _alloc(), use_dynamic_sampling=False, trainer="sync"
            )

    def test_a_non_sync_trainer_is_refused(self):
        with pytest.raises(ValueError, match="async"):
            validate_graded_allocation_config(
                _alloc(), use_dynamic_sampling=True, trainer="async"
            )

    def test_a_negative_exponent_is_refused(self):
        with pytest.raises(ValueError, match="variance_exponent"):
            validate_graded_allocation_config(
                _alloc(variance_exponent=-1.0),
                use_dynamic_sampling=True,
                trainer="sync",
            )

    def test_a_negative_floor_is_refused(self):
        with pytest.raises(ValueError, match="min_rollouts_per_prompt"):
            validate_graded_allocation_config(
                _alloc(min_rollouts_per_prompt=-1),
                use_dynamic_sampling=True,
                trainer="sync",
            )

    def test_a_non_integer_floor_is_refused(self):
        with pytest.raises(ValueError, match="min_rollouts_per_prompt"):
            validate_graded_allocation_config(
                _alloc(min_rollouts_per_prompt=1.5),
                use_dynamic_sampling=True,
                trainer="sync",
            )

    def test_enabled_helper_reads_the_flag(self):
        assert graded_allocation_enabled(_alloc()) is True
        assert graded_allocation_enabled(_alloc(enabled=False)) is False
        assert graded_allocation_enabled(None) is False


# ── In-pipeline, through dynamic_sampling


def _pipeline_batch(sizes: list[int]) -> BatchedDataDict:
    n = sum(sizes)
    return BatchedDataDict(
        {
            "total_reward": torch.arange(n, dtype=torch.float32),
            "idx": torch.arange(n),
            GRADED_ALLOCATION_GROUP_KEY: _group_ids(sizes),
        }
    )


def _run_pipeline(batch, std, cfg, ordinal=0):
    from dockyard_rl.algorithms.grpo import dynamic_sampling

    return dynamic_sampling(
        batch,
        std,
        torch.zeros_like(std),
        ordinal,
        cast(Any, cfg),
        _Timer(),
        batch_cache=None,
    )


class TestThroughDynamicSampling:
    """Sizes are chosen so the surplus is real: 3 groups of 4 = 12 rows survive
    the variance gate, and the train batch takes 4."""

    SIZES = [4, 4, 4]

    def _std(self, per_group):
        return _row_weights(self.SIZES, per_group)

    def _cfg(self, **alloc):
        return _master_config(
            num_prompts_per_step=2,
            num_generations_per_prompt=2,
            graded_allocation=_alloc(**alloc),
        )

    def test_disabled_is_byte_for_byte_head_truncation(self):
        batch = _pipeline_batch(self.SIZES)
        std = self._std([0.1, 0.9, 0.5])
        out, complete, _cache, metrics = _run_pipeline(
            batch,
            std,
            _master_config(
                num_prompts_per_step=2,
                num_generations_per_prompt=2,
                graded_allocation=_alloc(enabled=False),
            ),
        )
        assert complete is True
        assert out["idx"].tolist() == [0, 1, 2, 3]
        assert not any(k.startswith("graded_allocation") for k in metrics)

    def test_absent_config_is_head_truncation(self):
        batch = _pipeline_batch(self.SIZES)
        out, _c, _cache, _m = _run_pipeline(
            batch,
            self._std([0.1, 0.9, 0.5]),
            _master_config(num_prompts_per_step=2, num_generations_per_prompt=2),
        )
        assert out["idx"].tolist() == [0, 1, 2, 3]

    def test_enabled_moves_the_budget_off_arrival_order(self):
        batch = _pipeline_batch(self.SIZES)
        out, complete, _cache, metrics = _run_pipeline(
            batch, self._std([0.01, 0.98, 0.01]), self._cfg()
        )
        assert complete is True
        assert out.size == 4
        kept = out["idx"].tolist()
        # Group 1 (rows 4-7) carries almost all the variance; head truncation
        # would have kept rows 0-3 and never touched it.
        assert any(4 <= i <= 7 for i in kept)
        assert metrics["graded_allocation_rows_reallocated"] > 0

    def test_the_floor_keeps_every_surviving_group_in_the_batch(self):
        batch = _pipeline_batch(self.SIZES)
        out, _c, _cache, metrics = _run_pipeline(
            batch, self._std([0.01, 0.98, 0.01]), self._cfg()
        )
        kept = out["idx"].tolist()
        assert {i // 4 for i in kept} == {0, 1, 2}
        assert metrics["graded_allocation_groups_represented"] == 3.0

    def test_the_row_count_is_unchanged_so_the_step_costs_the_same(self):
        for exponent in (0.0, 1.0, 4.0):
            batch = _pipeline_batch(self.SIZES)
            out, _c, _cache, _m = _run_pipeline(
                batch,
                self._std([0.2, 0.5, 0.3]),
                self._cfg(variance_exponent=exponent),
            )
            assert out.size == 4

    def test_the_group_column_does_not_reach_training(self):
        batch = _pipeline_batch(self.SIZES)
        out, _c, _cache, _m = _run_pipeline(
            batch, self._std([0.2, 0.5, 0.3]), self._cfg()
        )
        assert GRADED_ALLOCATION_GROUP_KEY not in out

    def test_baseline_and_std_ride_the_selected_rows(self):
        batch = _pipeline_batch(self.SIZES)
        std = self._std([0.01, 0.98, 0.01])
        out, _c, _cache, _m = _run_pipeline(batch, std, self._cfg())
        for row, idx in enumerate(out["idx"].tolist()):
            assert float(out["std"][row]) == pytest.approx(float(std[idx]))

    def test_the_zero_variance_gate_still_runs_first(self):
        # Group 0 has zero std: it is dropped by the gate, not floored back in.
        batch = _pipeline_batch(self.SIZES)
        out, _c, _cache, _m = _run_pipeline(
            batch, self._std([0.0, 0.9, 0.5]), self._cfg()
        )
        assert all(i >= 4 for i in out["idx"].tolist())

    def test_a_missing_group_column_is_refused_rather_than_ignored(self):
        batch = _pipeline_batch(self.SIZES)
        batch.pop(GRADED_ALLOCATION_GROUP_KEY)
        with pytest.raises(ValueError, match=GRADED_ALLOCATION_GROUP_KEY):
            _run_pipeline(batch, self._std([0.2, 0.5, 0.3]), self._cfg())

    def test_an_incomplete_batch_is_untouched_by_the_allocator(self):
        # 4 surviving rows, 8 required: the step generates again instead.
        batch = _pipeline_batch([2, 2])
        out, complete, cache, metrics = _run_pipeline(
            batch,
            _row_weights([2, 2], [0.5, 0.5]),
            _master_config(
                num_prompts_per_step=4,
                num_generations_per_prompt=2,
                graded_allocation=_alloc(),
            ),
        )
        assert complete is False
        assert cache is not None
        assert not any(k.startswith("graded_allocation") for k in metrics)
        assert GRADED_ALLOCATION_GROUP_KEY in out

    def test_the_metrics_describe_the_allocation(self):
        batch = _pipeline_batch(self.SIZES)
        _out, _c, _cache, metrics = _run_pipeline(
            batch, self._std([0.01, 0.98, 0.01]), self._cfg()
        )
        assert metrics["graded_allocation_groups"] == 3.0
        assert metrics["graded_allocation_rows_per_group_min"] >= 1.0
        assert metrics["graded_allocation_rows_per_group_max"] <= 4.0
        assert metrics["dynamic_sampling_num_discarded_valid_samples"] == 8


class TestMetrics:
    def test_reallocated_counts_rows_truncation_would_have_dropped(self):
        gids = _group_ids([4, 4])
        metrics = graded_allocation_metrics(gids, [0, 1, 4, 5], budget=4)
        assert metrics["graded_allocation_rows_reallocated"] == 2.0

    def test_reallocated_is_zero_when_the_allocation_agrees_with_order(self):
        gids = _group_ids([4, 4])
        metrics = graded_allocation_metrics(gids, [0, 1, 2, 3], budget=4)
        assert metrics["graded_allocation_rows_reallocated"] == 0.0


# ── Wires, cut individually


class TestWiring:
    def _grpo_source(self):
        from dockyard_rl.algorithms import grpo

        return inspect.getsource(grpo)

    def test_setup_validates_the_config(self):
        from dockyard_rl.algorithms.grpo import setup

        assert "_validate_graded_allocation_compat(master_config)" in (
            inspect.getsource(setup)
        )

    def test_the_setup_validator_reads_the_real_dynamic_sampling_flag(self):
        # Behavioural, not textual: a marshaller that hardcodes the flag keeps
        # the keyword in its source and refuses nothing.
        from dockyard_rl.algorithms.grpo import _validate_graded_allocation_compat

        with pytest.raises(ValueError, match="use_dynamic_sampling"):
            _validate_graded_allocation_compat(
                cast(
                    Any,
                    _master_config(
                        use_dynamic_sampling=False, graded_allocation=_alloc()
                    ),
                )
            )

    def test_the_setup_validator_admits_a_valid_config(self):
        from dockyard_rl.algorithms.grpo import _validate_graded_allocation_compat

        _validate_graded_allocation_compat(
            cast(
                Any,
                _master_config(
                    use_dynamic_sampling=True, graded_allocation=_alloc()
                ),
            )
        )

    def test_the_setup_validator_is_inert_when_disabled(self):
        from dockyard_rl.algorithms.grpo import _validate_graded_allocation_compat

        _validate_graded_allocation_compat(
            cast(
                Any,
                _master_config(
                    use_dynamic_sampling=False,
                    graded_allocation=_alloc(enabled=False),
                ),
            )
        )

    def test_dynamic_sampling_calls_the_allocator(self):
        from dockyard_rl.algorithms.grpo import dynamic_sampling

        source = inspect.getsource(dynamic_sampling)
        assert "allocate_rollout_budget(" in source
        assert "graded_allocation_metrics(" in source
        assert f"pop({GRADED_ALLOCATION_GROUP_KEY}" in source.replace(
            "GRADED_ALLOCATION_GROUP_KEY", GRADED_ALLOCATION_GROUP_KEY
        ) or "pop(GRADED_ALLOCATION_GROUP_KEY" in source

    def test_the_sync_step_attaches_the_group_column_under_the_enabled_guard(self):
        # Structural, not textual: replacing the guard with `if False:` leaves
        # every substring in place while the column is never attached, so the
        # allocator refuses at the seam on a run that reads as configured.
        import ast

        tree = ast.parse(self._grpo_source())
        guarded = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            gates = {
                n.func.id
                for n in ast.walk(node.test)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            if "graded_allocation_enabled" not in gates:
                continue
            body = "\n".join(ast.unparse(stmt) for stmt in node.body)
            if "assign_group_ids" in body:
                guarded.append(body)

        assert guarded, (
            "The group-id column must be attached under a "
            "graded_allocation_enabled() guard in the sync train step."
        )
        assert any(
            "dynamic_sampling_num_gen_batches" in body for body in guarded
        ), "The generation-batch ordinal must reach assign_group_ids."

    def test_the_group_column_is_built_from_the_baseline_prompt_tensor(self):
        source = self._grpo_source()
        attach = source.split("assign_group_ids(", 1)[1][:200]
        # `input_ids` is the same flattened prompt tensor passed to
        # calculate_baseline_and_std_per_prompt above it.
        assert "input_ids" in attach

    def test_the_shipped_config_carries_the_block_disabled(self):
        import pathlib

        import yaml

        root = pathlib.Path(__file__).resolve().parents[2]
        cfg = yaml.safe_load(
            (root / "examples/configs/grpo_swe.yaml").read_text()
        )
        block = cfg["grpo"]["graded_allocation"]
        assert block["enabled"] is False
        assert block["min_rollouts_per_prompt"] == 1
        assert block["variance_exponent"] == 1.0
