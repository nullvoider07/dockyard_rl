"""Routing diversity as an allocation signal (F3b).

The load-bearing property here is a negative one: the signal must never reach a
reward term. Rewarding routing diversity is reward-hacking bait -- random routing
maximizes it and destroys the specialization that makes it worth measuring -- so
what makes it safe is structural, not a matter of tuning. `TestNeverPriced`
below is the test that enforces it, and it is the reason the signal lives in a
module that imports nothing from the reward path.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from dockyard_rl.algorithms.rollout_allocation import (
    GRADED_ALLOCATION_GROUP_KEY,
    allocate_rollout_budget,
)
from dockyard_rl.algorithms.routing_diversity import (
    ROUTING_DIVERSITY_KEY,
    routing_diversity_enabled,
    routing_diversity_from_message_log,
    routing_diversity_metrics,
    sample_routing_diversity,
    validate_routing_diversity_config,
)
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.models.generation.vllm.router_capture import MISSING_ROUTE_SENTINEL


class _Timer:
    @contextlib.contextmanager
    def time(self, _name):
        yield


def _routes(experts_per_position: list[list[int]], *, layers: int = 1) -> torch.Tensor:
    """[T, L, K] routes; every layer routes identically unless varied by caller."""
    t = torch.tensor(experts_per_position, dtype=torch.int32)  # [T, K]
    return t.unsqueeze(1).expand(t.shape[0], layers, t.shape[1]).clone()


def _turn(routes: torch.Tensor | None, *, n_tokens: int | None = None) -> dict:
    msg: dict[str, Any] = {"role": "assistant"}
    if routes is not None:
        msg["routed_experts"] = routes
        msg["token_ids"] = torch.zeros(routes.shape[0], dtype=torch.long)
    else:
        msg["token_ids"] = torch.zeros(n_tokens or 1, dtype=torch.long)
    return msg


def _cfg(**kw):
    cfg = {"enabled": True, "allocation_exponent": 1.0}
    cfg.update(kw)
    return cfg


# ── The statistic


class TestSampleRoutingDiversity:
    def test_a_rollout_with_no_routing_scores_zero_over_zero_positions(self):
        assert sample_routing_diversity([_turn(None, n_tokens=5)]) == (0.0, 0)

    def test_one_expert_everywhere_is_zero_entropy(self):
        # 5 positions, all routing to expert 3. The last is dropped as a
        # next-token artifact, leaving 4.
        routes = _routes([[3]] * 5)
        diversity, positions = sample_routing_diversity([_turn(routes)])
        assert positions == 4
        assert diversity == pytest.approx(0.0)

    def test_a_uniform_split_over_two_experts_is_ln_two(self):
        routes = _routes([[0], [1], [0], [1], [0]])
        diversity, positions = sample_routing_diversity([_turn(routes)])
        assert positions == 4
        assert diversity == pytest.approx(math.log(2), abs=1e-6)

    def test_a_uniform_split_over_four_experts_is_ln_four(self):
        routes = _routes([[0], [1], [2], [3], [0]])
        diversity, _p = sample_routing_diversity([_turn(routes)])
        assert diversity == pytest.approx(math.log(4), abs=1e-6)

    def test_wider_routing_scores_higher(self):
        narrow, _ = sample_routing_diversity([_turn(_routes([[0], [0], [0], [1], [0]]))])
        wide, _ = sample_routing_diversity([_turn(_routes([[0], [1], [2], [3], [0]]))])
        assert wide > narrow

    def test_the_final_position_is_dropped_as_a_next_token_artifact(self):
        # routed_experts_empty fills the trajectory's last position and all
        # padding with the identity route arange(top_k) -- a fabricated route.
        # Counting it would report diversity a concentrated rollout does not have.
        concentrated = _routes([[5, 5], [5, 5], [5, 5], [0, 1]])
        diversity, positions = sample_routing_diversity([_turn(concentrated)])
        assert positions == 3
        assert diversity == pytest.approx(0.0)

    def test_a_single_position_turn_contributes_nothing(self):
        assert sample_routing_diversity([_turn(_routes([[1]]))]) == (0.0, 0)

    def test_the_missing_route_sentinel_leaves_the_histogram(self):
        # -1 is not an expert. Counting it as one invents a category and
        # inflates entropy on exactly the batches where capture went wrong.
        clean = _routes([[7], [7], [7], [7], [7]])
        holed = _routes(
            [[7], [MISSING_ROUTE_SENTINEL], [7], [MISSING_ROUTE_SENTINEL], [7]]
        )
        assert sample_routing_diversity([_turn(clean)])[0] == pytest.approx(0.0)
        assert sample_routing_diversity([_turn(holed)])[0] == pytest.approx(0.0)

    def test_a_fully_missing_layer_does_not_drag_the_mean(self):
        routes = _routes([[MISSING_ROUTE_SENTINEL]] * 5)
        diversity, positions = sample_routing_diversity([_turn(routes)])
        assert positions == 4
        assert diversity == pytest.approx(0.0)

    def test_layers_are_averaged_not_pooled(self):
        # Layer 0 concentrated, layer 1 uniform over two experts. Pooling the
        # two histograms would report a different number than averaging their
        # entropies, and pooling is wrong: expert 3 in layer 0 is a different
        # parameter block from expert 3 in layer 1.
        stacked = torch.stack(
            [
                torch.tensor([[2], [2], [2], [2], [2]], dtype=torch.int32),
                torch.tensor([[0], [1], [0], [1], [0]], dtype=torch.int32),
            ],
            dim=1,
        )  # [T=5, L=2, K=1]
        diversity, _p = sample_routing_diversity([_turn(stacked)])
        assert diversity == pytest.approx(math.log(2) / 2, abs=1e-6)

    def test_multiple_turns_are_concatenated(self):
        first = _routes([[0], [0], [0]])
        second = _routes([[1], [1], [1]])
        both, positions = sample_routing_diversity([_turn(first), _turn(second)])
        assert positions == 4
        assert both == pytest.approx(math.log(2), abs=1e-6)

    def test_non_assistant_messages_without_routing_are_skipped(self):
        log = [_turn(None, n_tokens=9), _turn(_routes([[0], [1], [0], [1], [0]]))]
        diversity, positions = sample_routing_diversity(log)
        assert positions == 4
        assert diversity == pytest.approx(math.log(2), abs=1e-6)

    def test_entropy_is_length_robust_once_picks_exceed_experts(self):
        # The reason the statistic is entropy and not a distinct-expert count:
        # a count grows with length until it saturates, so it would re-rank
        # rollouts by length -- which F2 already prices.
        short = _routes([[0], [1]] * 8 + [[0]])
        long = _routes([[0], [1]] * 64 + [[0]])
        s, _ = sample_routing_diversity([_turn(short)])
        l, _ = sample_routing_diversity([_turn(long)])
        assert s == pytest.approx(l, abs=1e-6)


class TestFromMessageLog:
    def test_absent_routing_returns_none_not_a_zero_column(self):
        # A zero column would read as "every rollout is perfectly concentrated"
        # rather than "this quantity does not exist on a dense model".
        diversity, positions = routing_diversity_from_message_log(
            [[_turn(None, n_tokens=4)], [_turn(None, n_tokens=6)]]
        )
        assert diversity is None
        assert positions.tolist() == [0, 0]

    def test_one_row_per_sample_in_order(self):
        log = [
            [_turn(_routes([[0], [0], [0], [0], [0]]))],
            [_turn(_routes([[0], [1], [2], [3], [0]]))],
        ]
        diversity, positions = routing_diversity_from_message_log(log)
        assert diversity is not None
        assert diversity.shape == (2,)
        assert positions.tolist() == [4, 4]
        assert float(diversity[1]) > float(diversity[0])

    def test_a_partially_covered_batch_still_returns_a_column(self):
        log = [[_turn(None, n_tokens=3)], [_turn(_routes([[0], [1], [0], [1], [0]]))]]
        diversity, positions = routing_diversity_from_message_log(log)
        assert diversity is not None
        assert float(diversity[0]) == pytest.approx(0.0)
        assert positions.tolist() == [0, 4]

    def test_metrics_report_absence_distinctly(self):
        assert routing_diversity_metrics(None, torch.zeros(2, dtype=torch.int64)) == {
            "routing_diversity_present": 0.0
        }

    def test_metrics_report_the_spread_and_the_shortest_rollout(self):
        metrics = routing_diversity_metrics(
            torch.tensor([0.0, 2.0]), torch.tensor([4, 90])
        )
        assert metrics["routing_diversity_present"] == 1.0
        assert metrics["routing_diversity_mean"] == pytest.approx(1.0)
        assert metrics["routing_diversity_max"] == pytest.approx(2.0)
        assert metrics["routing_diversity_min_positions"] == 4.0


# ── The allocation gate


def _group_ids(sizes: list[int]) -> torch.Tensor:
    out: list[int] = []
    for gid, n in enumerate(sizes):
        out.extend([gid] * n)
    return torch.tensor(out, dtype=torch.int64)


def _per_row(sizes: list[int], per_group: list[float]) -> torch.Tensor:
    out: list[float] = []
    for n, v in zip(sizes, per_group):
        out.extend([v] * n)
    return torch.tensor(out, dtype=torch.float32)


def _counts(group_ids: torch.Tensor, kept: list[int]) -> dict[int, int]:
    ids = group_ids.tolist()
    counts: dict[int, int] = {}
    for r in kept:
        counts[int(ids[r])] = counts.get(int(ids[r]), 0) + 1
    return counts


class TestAllocationGate:
    SIZES = [4, 4]

    def test_exponent_zero_is_byte_for_byte_f3a(self):
        gids = _group_ids(self.SIZES)
        std = _per_row(self.SIZES, [0.5, 0.5])
        without = allocate_rollout_budget(gids, std, 4)
        with_signal = allocate_rollout_budget(
            gids,
            std,
            4,
            diversity=_per_row(self.SIZES, [0.01, 9.0]),
            diversity_exponent=0.0,
        )
        assert without == with_signal

    def test_a_none_signal_is_ignored(self):
        gids = _group_ids(self.SIZES)
        std = _per_row(self.SIZES, [0.5, 0.5])
        assert allocate_rollout_budget(
            gids, std, 4, diversity=None, diversity_exponent=4.0
        ) == allocate_rollout_budget(gids, std, 4)

    def test_diversity_moves_budget_when_variance_ties(self):
        gids = _group_ids(self.SIZES)
        std = _per_row(self.SIZES, [0.5, 0.5])
        kept = allocate_rollout_budget(
            gids,
            std,
            4,
            diversity=_per_row(self.SIZES, [0.2, 2.0]),
            diversity_exponent=2.0,
        )
        counts = _counts(gids, kept)
        assert counts[1] > counts[0]

    def test_variance_still_leads_when_diversity_is_flat(self):
        gids = _group_ids(self.SIZES)
        kept = allocate_rollout_budget(
            gids,
            _per_row(self.SIZES, [0.05, 0.95]),
            4,
            diversity=_per_row(self.SIZES, [1.0, 1.0]),
            diversity_exponent=3.0,
        )
        assert _counts(gids, kept)[1] > _counts(gids, kept)[0]

    def test_an_all_zero_signal_does_not_produce_nan_weights(self):
        gids = _group_ids(self.SIZES)
        std = _per_row(self.SIZES, [0.4, 0.6])
        kept = allocate_rollout_budget(
            gids,
            std,
            4,
            diversity=_per_row(self.SIZES, [0.0, 0.0]),
            diversity_exponent=2.0,
        )
        assert len(kept) == 4
        assert kept == allocate_rollout_budget(gids, std, 4)

    def test_the_budget_is_still_exact_under_the_gate(self):
        sizes = [3, 5, 4]
        gids = _group_ids(sizes)
        for exponent in (0.0, 1.0, 5.0):
            kept = allocate_rollout_budget(
                gids,
                _per_row(sizes, [0.3, 0.6, 0.1]),
                7,
                diversity=_per_row(sizes, [0.5, 3.0, 1.0]),
                diversity_exponent=exponent,
            )
            assert len(kept) == 7

    def test_the_allocation_is_invariant_to_the_signal_scale(self):
        # Why the signal ships in raw nats with no normalizer: every consumer
        # of the weight vector either divides by the active total
        # (`_apportion`) or only ranks (the floor squeeze), so multiplying the
        # whole diversity column by a positive constant cannot move one row.
        # A normalizer would therefore be code that reads as load-bearing and
        # is not -- which is exactly what an earlier revision shipped.
        sizes = [4, 4, 3]
        gids = _group_ids(sizes)
        std = _per_row(sizes, [0.4, 0.6, 0.5])
        base = _per_row(sizes, [0.5, 2.0, 1.25])
        reference = allocate_rollout_budget(
            gids, std, 6, diversity=base, diversity_exponent=2.0
        )
        for scale in (0.001, 7.0, 1000.0):
            assert (
                allocate_rollout_budget(
                    gids, std, 6, diversity=base * scale, diversity_exponent=2.0
                )
                == reference
            )

    def test_the_all_zero_guard_preserves_the_variance_grading(self):
        # Not scale-invariance: an all-zero column makes every factor zero, the
        # apportionment sees a zero total and degrades to round-robin, and F3a's
        # variance grading is lost. The guard is the only thing stopping that.
        sizes = [4, 4]
        gids = _group_ids(sizes)
        std = _per_row(sizes, [0.05, 0.95])
        kept = allocate_rollout_budget(
            gids,
            std,
            4,
            min_rollouts_per_prompt=0,
            diversity=_per_row(sizes, [0.0, 0.0]),
            diversity_exponent=2.0,
        )
        counts = _counts(gids, kept)
        assert counts.get(1, 0) > counts.get(0, 0)

    def test_the_floor_squeeze_ranks_by_the_graded_weight(self):
        # Floors of 2 over three groups want 6 rows against a budget of 4.
        # Group 2 has the least variance and by far the most diversity; with the
        # gate on it must outrank group 0.
        sizes = [3, 3, 3]
        gids = _group_ids(sizes)
        kept = allocate_rollout_budget(
            gids,
            _per_row(sizes, [0.5, 0.9, 0.4]),
            4,
            min_rollouts_per_prompt=2,
            diversity=_per_row(sizes, [0.1, 0.1, 9.0]),
            diversity_exponent=3.0,
        )
        counts = _counts(gids, kept)
        assert len(kept) == 4
        assert counts.get(2, 0) == 2
        assert counts.get(0, 0) == 0


# ── Configuration refusals


class TestValidation:
    def test_disabled_is_inert(self):
        validate_routing_diversity_config(
            _cfg(enabled=False),
            router_replay_enabled=False,
            graded_allocation_enabled=False,
            trainer="async",
        )

    def test_absent_is_inert(self):
        validate_routing_diversity_config(
            None,
            router_replay_enabled=False,
            graded_allocation_enabled=False,
            trainer="async",
        )

    def test_router_replay_off_is_refused(self):
        with pytest.raises(ValueError, match="router_replay"):
            validate_routing_diversity_config(
                _cfg(),
                router_replay_enabled=False,
                graded_allocation_enabled=True,
                trainer="sync",
            )

    def test_no_consumer_is_refused(self):
        with pytest.raises(ValueError, match="graded_allocation"):
            validate_routing_diversity_config(
                _cfg(),
                router_replay_enabled=True,
                graded_allocation_enabled=False,
                trainer="sync",
            )

    def test_a_non_sync_trainer_is_refused(self):
        with pytest.raises(ValueError, match="async"):
            validate_routing_diversity_config(
                _cfg(),
                router_replay_enabled=True,
                graded_allocation_enabled=True,
                trainer="async",
            )

    def test_a_negative_exponent_is_refused(self):
        with pytest.raises(ValueError, match="allocation_exponent"):
            validate_routing_diversity_config(
                _cfg(allocation_exponent=-1.0),
                router_replay_enabled=True,
                graded_allocation_enabled=True,
                trainer="sync",
            )

    def test_enabled_helper_reads_the_flag(self):
        assert routing_diversity_enabled(_cfg()) is True
        assert routing_diversity_enabled(_cfg(enabled=False)) is False
        assert routing_diversity_enabled(None) is False


class TestSetupMarshalling:
    def _master(self, *, router_replay, allocation, diversity):
        return SimpleNamespace(
            grpo={
                "num_prompts_per_step": 2,
                "num_generations_per_prompt": 2,
                "use_dynamic_sampling": True,
                "graded_allocation": {"enabled": allocation},
                "routing_diversity": diversity,
            },
            policy={"router_replay": {"enabled": router_replay}},
        )

    def test_the_marshaller_reads_the_real_router_replay_flag(self):
        from dockyard_rl.algorithms.grpo import _validate_routing_diversity_compat

        with pytest.raises(ValueError, match="router_replay"):
            _validate_routing_diversity_compat(
                cast(
                    Any,
                    self._master(
                        router_replay=False, allocation=True, diversity=_cfg()
                    ),
                )
            )

    def test_the_marshaller_reads_the_real_allocation_flag(self):
        from dockyard_rl.algorithms.grpo import _validate_routing_diversity_compat

        with pytest.raises(ValueError, match="graded_allocation"):
            _validate_routing_diversity_compat(
                cast(
                    Any,
                    self._master(
                        router_replay=True, allocation=False, diversity=_cfg()
                    ),
                )
            )

    def test_a_valid_config_is_admitted(self):
        from dockyard_rl.algorithms.grpo import _validate_routing_diversity_compat

        _validate_routing_diversity_compat(
            cast(
                Any,
                self._master(router_replay=True, allocation=True, diversity=_cfg()),
            )
        )

    def test_disabled_is_inert_even_with_everything_else_off(self):
        from dockyard_rl.algorithms.grpo import _validate_routing_diversity_compat

        _validate_routing_diversity_compat(
            cast(
                Any,
                self._master(
                    router_replay=False,
                    allocation=False,
                    diversity=_cfg(enabled=False),
                ),
            )
        )


# ── In-pipeline, through dynamic_sampling


def _master_config(**grpo):
    base: dict[str, Any] = {
        "num_prompts_per_step": 2,
        "num_generations_per_prompt": 2,
        "use_dynamic_sampling": True,
        "dynamic_sampling_max_gen_batches": 10,
        "graded_allocation": {
            "enabled": True,
            "min_rollouts_per_prompt": 1,
            "variance_exponent": 1.0,
        },
    }
    base.update(grpo)
    return SimpleNamespace(grpo=base)


def _pipeline_batch(sizes: list[int], diversity: list[float] | None) -> BatchedDataDict:
    n = sum(sizes)
    data: dict[str, Any] = {
        "total_reward": torch.arange(n, dtype=torch.float32),
        "idx": torch.arange(n),
        GRADED_ALLOCATION_GROUP_KEY: _group_ids(sizes),
    }
    if diversity is not None:
        data[ROUTING_DIVERSITY_KEY] = _per_row(sizes, diversity)
    return BatchedDataDict(data)


def _run(batch, std, cfg):
    from dockyard_rl.algorithms.grpo import dynamic_sampling

    return dynamic_sampling(
        batch,
        std,
        torch.zeros_like(std),
        0,
        cast(Any, cfg),
        _Timer(),
        batch_cache=None,
    )


class TestThroughDynamicSampling:
    SIZES = [4, 4, 4]

    def test_the_signal_column_never_reaches_training(self):
        batch = _pipeline_batch(self.SIZES, [1.0, 2.0, 3.0])
        out, _c, _cache, _m = _run(
            batch,
            _per_row(self.SIZES, [0.3, 0.3, 0.3]),
            _master_config(routing_diversity=_cfg()),
        )
        assert ROUTING_DIVERSITY_KEY not in out
        assert GRADED_ALLOCATION_GROUP_KEY not in out

    def test_the_signal_changes_which_rows_train(self):
        std = _per_row(self.SIZES, [0.3, 0.3, 0.3])
        flat, _c, _cache, _m = _run(
            _pipeline_batch(self.SIZES, [1.0, 1.0, 1.0]),
            std,
            _master_config(routing_diversity=_cfg(allocation_exponent=4.0)),
        )
        skewed, _c2, _cache2, _m2 = _run(
            _pipeline_batch(self.SIZES, [0.1, 0.1, 5.0]),
            std,
            _master_config(routing_diversity=_cfg(allocation_exponent=4.0)),
        )
        assert flat["idx"].tolist() != skewed["idx"].tolist()
        assert sum(1 for i in skewed["idx"].tolist() if i >= 8) > sum(
            1 for i in flat["idx"].tolist() if i >= 8
        )

    def test_disabled_leaves_f3a_untouched(self):
        std = _per_row(self.SIZES, [0.1, 0.9, 0.5])
        with_column, _c, _cache, _m = _run(
            _pipeline_batch(self.SIZES, [0.1, 0.1, 9.0]),
            std,
            _master_config(routing_diversity=_cfg(enabled=False)),
        )
        without, _c2, _cache2, _m2 = _run(
            _pipeline_batch(self.SIZES, None), std, _master_config()
        )
        assert with_column["idx"].tolist() == without["idx"].tolist()

    def test_an_absent_column_under_an_enabled_config_is_inert(self):
        # A dense model with the signal switched on: nothing carries routing, so
        # routing_diversity_from_message_log returns None and no column is
        # attached. The allocation must degrade to F3a, not fail.
        std = _per_row(self.SIZES, [0.1, 0.9, 0.5])
        out, _c, _cache, _m = _run(
            _pipeline_batch(self.SIZES, None),
            std,
            _master_config(routing_diversity=_cfg(allocation_exponent=3.0)),
        )
        baseline, _c2, _cache2, _m2 = _run(
            _pipeline_batch(self.SIZES, None), std, _master_config()
        )
        assert out["idx"].tolist() == baseline["idx"].tolist()


# ── The negative property: never priced


class TestNeverPriced:
    """The signal must not reach a reward term.

    Rewarding it would make random routing optimal. Nothing about the value is
    safe -- what is safe is that no gradient path touches it, so these tests are
    about imports and assignments, not numbers."""

    def test_the_module_imports_nothing_from_the_reward_path(self):
        from dockyard_rl.algorithms import routing_diversity

        tree = ast.parse(inspect.getsource(routing_diversity))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        forbidden = {
            "dockyard_rl.algorithms.reward_functions",
            "dockyard_rl.algorithms.effort_reward",
            "dockyard_rl.algorithms.advantage_estimator",
        }
        assert not (imported & forbidden), (
            f"routing_diversity must not import the reward path: "
            f"{sorted(imported & forbidden)}"
        )

    def test_no_reward_module_reads_the_column(self):
        from dockyard_rl.algorithms import (
            advantage_estimator,
            effort_reward,
            reward_functions,
        )

        for module in (reward_functions, effort_reward, advantage_estimator):
            source = inspect.getsource(module)
            assert ROUTING_DIVERSITY_KEY not in source, (
                f"{module.__name__} references {ROUTING_DIVERSITY_KEY!r}; the "
                "routing-diversity signal must never enter a reward term."
            )

    def test_the_column_is_never_assigned_into_a_reward_key(self):
        from dockyard_rl.algorithms import grpo

        tree = ast.parse(inspect.getsource(grpo))
        reward_keys = {
            "total_reward",
            "unshaped_total_reward",
            "filtered_reward",
            "rewards",
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            rendered = ast.unparse(node)
            if ROUTING_DIVERSITY_KEY not in rendered and (
                "ROUTING_DIVERSITY_KEY" not in rendered
            ):
                continue
            for target in node.targets:
                text = ast.unparse(target)
                assert not any(key in text for key in reward_keys), (
                    f"routing diversity flows into a reward key: {rendered}"
                )

    def test_the_grpo_reward_pipeline_runs_before_the_signal_is_built(self):
        # Ordering is the structural guarantee: every reward writer has already
        # run by the time the column exists, so none of them can read it.
        from dockyard_rl.algorithms import grpo

        source = inspect.getsource(grpo)
        effort = source.index("repeated_batch = apply_effort_reward(")
        signal = source.index("routing_diversity_from_message_log(")
        assert effort < signal, (
            "the routing-diversity column must be built after the reward "
            "pipeline, so no reward term can consume it"
        )

    def test_the_allocator_is_the_only_consumer(self):
        from dockyard_rl.algorithms import grpo

        source = inspect.getsource(grpo)
        reads = source.count("ROUTING_DIVERSITY_KEY")
        # import, attach, allocator hand-off, pop -- and nothing else.
        assert reads == 4, (
            f"expected exactly 4 ROUTING_DIVERSITY_KEY sites (import, attach, "
            f"allocate, pop), found {reads}; a new consumer needs its own review"
        )


# ── Wires


class TestWiring:
    def test_setup_validates_the_config(self):
        from dockyard_rl.algorithms.grpo import setup

        assert "_validate_routing_diversity_compat(master_config)" in (
            inspect.getsource(setup)
        )

    def test_the_attach_is_guarded_by_the_enabled_flag(self):
        from dockyard_rl.algorithms import grpo

        tree = ast.parse(inspect.getsource(grpo))
        guarded = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            gates = {
                n.func.id
                for n in ast.walk(node.test)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            if "routing_diversity_enabled" not in gates:
                continue
            body = "\n".join(ast.unparse(stmt) for stmt in node.body)
            if "routing_diversity_from_message_log" in body:
                guarded.append(body)
        assert guarded, (
            "the routing-diversity column must be attached under a "
            "routing_diversity_enabled() guard"
        )
        assert any("message_log" in body for body in guarded)

    def test_dynamic_sampling_hands_the_signal_to_the_allocator(self):
        from dockyard_rl.algorithms.grpo import dynamic_sampling

        source = inspect.getsource(dynamic_sampling)
        assert "diversity=filtered_repeated_batch.get(" in source
        assert "diversity_exponent=(" in source

    def test_the_shipped_config_carries_the_block_disabled(self):
        import pathlib

        import yaml

        root = pathlib.Path(__file__).resolve().parents[2]
        cfg = yaml.safe_load((root / "examples/configs/grpo_swe.yaml").read_text())
        block = cfg["grpo"]["routing_diversity"]
        assert block["enabled"] is False
        assert block["allocation_exponent"] == 0.0
