"""Teacher-panel bursting (F3c).

The property that matters is cost: a panel exists to be *not* paid for on most
rows. So the tests below check what was scored and by whom, not only that the
numbers come out combined -- an implementation that quietly scores every row
with every teacher produces correct logprobs and defeats the entire feature.
"""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from dockyard_rl.algorithms.teacher_panel import (
    COMBINE_MODES,
    combine_panel_logprobs,
    mean_valid_logprob,
    select_escalation_rows,
    teacher_panel_cfg,
    teacher_panel_enabled,
    teacher_panel_metrics,
    validate_teacher_panel_config,
)
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict


def _panel(**kw):
    cfg = {
        "enabled": True,
        "panel_aliases": ["t2"],
        "escalation_fraction": 0.5,
        "combine": "mixture",
    }
    cfg.update(kw)
    return cfg


def _opd(**kw):
    base = {
        "teacher_model_by_agent_name": {"t1": "m1", "t2": "m2"},
        "default_teacher_alias": "t1",
    }
    base.update(kw)
    return base


# ── The escalation signal


class TestMeanValidLogprob:
    def test_padding_is_excluded(self):
        # Row 0 is 2 tokens then padding; row 1 is 4 tokens. A plain row mean
        # would rank these by how much padding they carry, i.e. by length.
        logprobs = torch.tensor([[-1.0, -1.0, -9.0, -9.0], [-2.0] * 4])
        out = mean_valid_logprob(logprobs, torch.tensor([2, 4]))
        assert out.tolist() == pytest.approx([-1.0, -2.0])

    def test_absent_lengths_fall_back_to_the_whole_row(self):
        logprobs = torch.tensor([[-1.0, -3.0]])
        assert mean_valid_logprob(logprobs, None).tolist() == pytest.approx([-2.0])

    def test_a_zero_length_row_does_not_divide_by_zero(self):
        out = mean_valid_logprob(torch.tensor([[-1.0, -2.0]]), torch.tensor([0]))
        assert out.tolist() == [0.0]

    def test_lengths_beyond_the_sequence_are_clamped(self):
        out = mean_valid_logprob(torch.tensor([[-1.0, -3.0]]), torch.tensor([99]))
        assert out.tolist() == pytest.approx([-2.0])

    def test_a_non_2d_input_is_refused(self):
        with pytest.raises(ValueError, match=r"\[B, S\]"):
            mean_valid_logprob(torch.zeros(3), None)


class TestSelectEscalationRows:
    def test_the_least_likely_rows_are_chosen(self):
        scores = torch.tensor([-0.1, -9.0, -0.2, -8.0])
        assert select_escalation_rows(scores, 0.5) == [1, 3]

    def test_indices_come_back_ascending(self):
        scores = torch.tensor([-9.0, -0.1, -8.0, -0.2])
        assert select_escalation_rows(scores, 0.5) == [0, 2]

    def test_the_fraction_is_an_upper_bound_not_a_target(self):
        # 3 rows at 0.5 rounds down to 1, so the burst can never cost more than
        # the configured share.
        assert len(select_escalation_rows(torch.tensor([-1.0, -2.0, -3.0]), 0.5)) == 1

    def test_zero_fraction_bursts_nothing(self):
        assert select_escalation_rows(torch.tensor([-1.0, -2.0]), 0.0) == []

    def test_a_fraction_too_small_to_reach_one_row_bursts_nothing(self):
        assert select_escalation_rows(torch.tensor([-1.0, -2.0]), 0.1) == []

    def test_a_full_fraction_takes_everything(self):
        assert select_escalation_rows(torch.tensor([-1.0, -2.0]), 1.0) == [0, 1]

    def test_an_empty_batch_is_empty(self):
        assert select_escalation_rows(torch.tensor([]), 1.0) == []

    def test_ties_break_deterministically_on_index(self):
        scores = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        assert select_escalation_rows(scores, 0.5) == [0, 1]


# ── Combining


class TestCombine:
    def test_a_single_teacher_passes_through(self):
        only = torch.tensor([[-1.0, -2.0]])
        assert torch.equal(combine_panel_logprobs([only]), only)

    def test_mixture_is_the_logprob_of_the_average_distribution(self):
        # Two teachers giving p=0.5 and p=0.5 -> mixture p=0.5.
        lp = math.log(0.5)
        out = combine_panel_logprobs(
            [torch.tensor([[lp]]), torch.tensor([[lp]])], "mixture"
        )
        assert float(out[0, 0]) == pytest.approx(lp)

    def test_mixture_averages_probabilities_not_logprobs(self):
        # p = 0.8 and p = 0.2 -> mixture 0.5. The geometric mean would be 0.4.
        a, b = math.log(0.8), math.log(0.2)
        mixture = combine_panel_logprobs(
            [torch.tensor([[a]]), torch.tensor([[b]])], "mixture"
        )
        assert math.exp(float(mixture[0, 0])) == pytest.approx(0.5)

    def test_geometric_is_the_mean_of_logprobs(self):
        a, b = math.log(0.8), math.log(0.2)
        geo = combine_panel_logprobs(
            [torch.tensor([[a]]), torch.tensor([[b]])], "geometric"
        )
        assert math.exp(float(geo[0, 0])) == pytest.approx(0.4)

    def test_one_dissenting_teacher_cannot_veto_under_mixture(self):
        # The reason mixture is the default: a panel is specialists, most of
        # whom are irrelevant to a given trajectory. Under a product of experts
        # a single confident dissenter drags the target to near zero.
        confident = math.log(0.9)
        dissenter = math.log(1e-8)
        mixture = combine_panel_logprobs(
            [torch.tensor([[confident]]), torch.tensor([[dissenter]])], "mixture"
        )
        geo = combine_panel_logprobs(
            [torch.tensor([[confident]]), torch.tensor([[dissenter]])], "geometric"
        )
        assert math.exp(float(mixture[0, 0])) == pytest.approx(0.45, abs=1e-3)
        assert math.exp(float(geo[0, 0])) < 1e-4

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="share a shape"):
            combine_panel_logprobs([torch.zeros(1, 2), torch.zeros(1, 3)])

    def test_an_empty_panel_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            combine_panel_logprobs([])

    def test_an_unknown_combine_mode_is_refused(self):
        with pytest.raises(ValueError, match="combine must be"):
            combine_panel_logprobs([torch.zeros(1, 2)] * 2, "median")

    def test_the_mode_set_is_what_validation_accepts(self):
        assert COMBINE_MODES == {"mixture", "geometric"}


# ── Configuration refusals


class TestValidation:
    def test_disabled_is_inert(self):
        validate_teacher_panel_config(
            _opd(teacher_panel=_panel(enabled=False)),
            available_aliases=[],
            trainer="sync",
        )

    def test_absent_is_inert(self):
        validate_teacher_panel_config(_opd(), available_aliases=[], trainer="sync")

    def test_a_sync_trainer_is_refused(self):
        with pytest.raises(ValueError, match="sync"):
            validate_teacher_panel_config(
                _opd(teacher_panel=_panel()),
                available_aliases=["t1", "t2"],
                trainer="sync",
            )

    def test_an_unbuilt_panel_alias_is_refused(self):
        with pytest.raises(ValueError, match="ghost"):
            validate_teacher_panel_config(
                _opd(teacher_panel=_panel(panel_aliases=["ghost"])),
                available_aliases=["t1", "t2"],
                trainer="async",
            )

    def test_an_empty_panel_is_refused(self):
        with pytest.raises(ValueError, match="no panel_aliases"):
            validate_teacher_panel_config(
                _opd(teacher_panel=_panel(panel_aliases=[])),
                available_aliases=["t1"],
                trainer="async",
            )

    def test_an_out_of_range_fraction_is_refused(self):
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError, match="escalation_fraction"):
                validate_teacher_panel_config(
                    _opd(teacher_panel=_panel(escalation_fraction=bad)),
                    available_aliases=["t1", "t2"],
                    trainer="async",
                )

    def test_an_unknown_combine_mode_is_refused(self):
        with pytest.raises(ValueError, match="combine"):
            validate_teacher_panel_config(
                _opd(teacher_panel=_panel(combine="median")),
                available_aliases=["t1", "t2"],
                trainer="async",
            )

    def test_a_valid_panel_is_admitted(self):
        validate_teacher_panel_config(
            _opd(teacher_panel=_panel()),
            available_aliases=["t1", "t2"],
            trainer="async",
        )

    def test_the_enabled_helper_reads_a_pydantic_block_too(self):
        from dockyard_rl.algorithms.opd import (
            OnPolicyDistillationConfig,
            TeacherPanelConfig,
        )

        cfg = OnPolicyDistillationConfig(
            enabled=True,
            teacher_panel=TeacherPanelConfig(
                enabled=True, panel_aliases=["t2"], escalation_fraction=0.25
            ),
        )
        assert teacher_panel_enabled(cfg) is True
        assert teacher_panel_cfg(cfg)["escalation_fraction"] == 0.25

    def test_the_enabled_helper_is_false_on_a_bare_config(self):
        assert teacher_panel_enabled(None) is False
        assert teacher_panel_enabled({}) is False
        assert teacher_panel_enabled(_opd()) is False


class TestMetrics:
    def test_saving_is_measured_against_a_full_panel(self):
        # 10 rows, panel of 2, 2 rows escalated: 10 + 2*2 = 14 teacher-rows
        # against a full panel's 10*3 = 30.
        m = teacher_panel_metrics([0, 1], batch_size=10, panel_size=2)
        assert m["teacher_panel_rows_scored"] == 14.0
        assert m["teacher_panel_rows_saved_ratio"] == pytest.approx(1 - 14 / 30)
        assert m["teacher_panel_escalated_ratio"] == pytest.approx(0.2)

    def test_no_escalation_reports_the_base_cost(self):
        m = teacher_panel_metrics([], batch_size=8, panel_size=3)
        assert m["teacher_panel_escalated_rows"] == 0.0
        assert m["teacher_panel_rows_scored"] == 8.0

    def test_the_escalated_rows_are_the_more_surprising_ones(self):
        scores = torch.tensor([-0.1, -9.0, -0.2, -8.0])
        m = teacher_panel_metrics([1, 3], 4, 1, scores)
        assert (
            m["teacher_panel_escalated_surprise_mean"]
            < m["teacher_panel_batch_surprise_mean"]
        )

    def test_an_empty_batch_does_not_divide_by_zero(self):
        m = teacher_panel_metrics([], batch_size=0, panel_size=0)
        assert m["teacher_panel_escalated_ratio"] == 0.0
        assert m["teacher_panel_rows_saved_ratio"] == 0.0


# ── In-pipeline, through the collector


class _CountingScorer:
    requires_prepadded_input = False
    sequence_pad_multiple = 1

    def __init__(self, alias, value):
        self.alias = alias
        self.value = value
        self.rows_scored: list[int] = []
        self.calls = 0

    def score(self, data):
        self.calls += 1
        ids = data["input_ids"]
        self.rows_scored.append(int(ids.shape[0]))
        return torch.full(ids.shape, self.value, dtype=torch.float32)

    def shutdown(self):
        return True


def _collector(panel_cfg, *, base_value=-1.0, panel_values=(-0.5,)):
    from dockyard_rl.algorithms.async_utils import trajectory_collector as tc

    col = tc.AsyncTrajectoryCollectorImpl.__new__(tc.AsyncTrajectoryCollectorImpl)
    base = _CountingScorer("t1", base_value)
    scorers: dict[str, Any] = {"t1": base}
    aliases = {"t1": "t1"}
    mapping = {"t1": "m1"}
    panel_scorers = []
    for i, value in enumerate(panel_values):
        alias = f"p{i}"
        s = _CountingScorer(alias, value)
        scorers[alias] = s
        aliases[alias] = alias
        mapping[alias] = f"pm{i}"
        panel_scorers.append(s)
    col._teacher_worker_groups = scorers
    col._alias_to_group_alias = aliases
    col._teacher_panel_metrics = {}
    col._on_policy_distillation_cfg = {
        "teacher_model_by_agent_name": mapping,
        "default_teacher_alias": "t1",
        "teacher_panel": panel_cfg,
    }
    return col, base, panel_scorers


def _score(col, batch=4, seq=3, lengths=None):
    input_ids = torch.arange(batch * seq).reshape(batch, seq)
    refs = [{"name": "t1"}] * batch
    return col._compute_teacher_logprobs(
        input_ids,
        refs,
        input_lengths=lengths if lengths is not None else torch.full((batch,), seq),
    )


class TestThroughTheCollector:
    def test_no_panel_scores_every_row_once(self):
        col, base, _p = _collector(None)
        result, _t = _score(col)
        assert base.calls == 1
        assert base.rows_scored == [4]
        assert result.shape == (4, 3)

    def test_no_panel_stamps_no_panel_metrics(self):
        # Without the enabled check the burst falls through to an empty
        # escalation and still stamps its metrics, so every non-panel OPD run
        # would grow teacher_panel_* keys describing a panel it does not have.
        col, _base, _panel = _collector(None)
        _score(col)
        assert col._teacher_panel_metrics == {}

    def test_a_disabled_panel_stamps_no_panel_metrics(self):
        col, _base, _panel = _collector(
            {"enabled": False, "panel_aliases": ["p0"], "escalation_fraction": 0.5}
        )
        _score(col)
        assert col._teacher_panel_metrics == {}

    def test_a_disabled_panel_is_byte_for_byte_the_base_pass(self):
        off, base_off, panel_off = _collector({"enabled": False, "panel_aliases": ["p0"]})
        none, base_none, _ = _collector(None)
        a, _ = _score(off)
        b, _ = _score(none)
        assert torch.equal(a, b)
        assert panel_off[0].calls == 0

    def test_the_panel_scores_only_the_escalated_rows(self):
        col, base, panel = _collector(
            {
                "enabled": True,
                "panel_aliases": ["p0"],
                "escalation_fraction": 0.5,
                "combine": "mixture",
            }
        )
        _score(col)
        assert base.rows_scored == [4]
        # Half of four rows, not four: this is the whole point of bursting.
        assert panel[0].rows_scored == [2]

    def test_unescalated_rows_keep_the_base_teacher_verbatim(self):
        col, _base, _panel = _collector(
            {
                "enabled": True,
                "panel_aliases": ["p0"],
                "escalation_fraction": 0.5,
                "combine": "mixture",
            },
            base_value=-1.0,
            panel_values=(-0.5,),
        )
        # Make rows 0 and 1 the surprising ones via shorter valid lengths?
        # Simpler: all rows tie, so ties break on index -> rows 0, 1 escalate.
        result, _t = _score(col)
        combined = math.log((math.exp(-1.0) + math.exp(-0.5)) / 2)
        assert result[0].tolist() == pytest.approx([combined] * 3)
        assert result[2].tolist() == pytest.approx([-1.0] * 3)

    def test_zero_fraction_calls_no_panel_teacher(self):
        col, _base, panel = _collector(
            {"enabled": True, "panel_aliases": ["p0"], "escalation_fraction": 0.0}
        )
        _score(col)
        assert panel[0].calls == 0

    def test_metrics_are_stamped_even_when_nothing_escalates(self):
        col, _base, _panel = _collector(
            {"enabled": True, "panel_aliases": ["p0"], "escalation_fraction": 0.0}
        )
        _score(col)
        assert col._teacher_panel_metrics["teacher_panel_escalated_rows"] == 0.0

    def test_metrics_report_the_burst(self):
        col, _base, _panel = _collector(
            {"enabled": True, "panel_aliases": ["p0"], "escalation_fraction": 0.5}
        )
        _score(col)
        m = col._teacher_panel_metrics
        assert m["teacher_panel_escalated_rows"] == 2.0
        assert m["teacher_panel_rows_scored"] == 6.0

    def test_a_multi_member_panel_scores_each_member_once(self):
        col, _base, panel = _collector(
            {"enabled": True, "panel_aliases": ["p0", "p1"], "escalation_fraction": 0.5},
            panel_values=(-0.5, -0.25),
        )
        _score(col)
        assert [s.rows_scored for s in panel] == [[2], [2]]

    def test_panel_members_sharing_a_group_are_scored_once(self):
        # Two aliases deduplicated onto one worker group must not score the same
        # rows twice -- that is the cost the burst exists to avoid.
        col, _base, panel = _collector(
            {"enabled": True, "panel_aliases": ["p0", "p1"], "escalation_fraction": 0.5},
            panel_values=(-0.5, -0.25),
        )
        col._alias_to_group_alias["p1"] = "p0"
        _score(col)
        assert panel[0].calls == 1
        assert panel[1].calls == 0

    def test_a_panel_alias_with_no_worker_group_bursts_nothing(self):
        col, _base, _panel = _collector(
            {"enabled": True, "panel_aliases": ["ghost"], "escalation_fraction": 0.5}
        )
        result, _t = _score(col)
        assert result[0].tolist() == pytest.approx([-1.0] * 3)

    def test_the_surprise_signal_drives_which_rows_burst(self):
        col, _base, panel = _collector(
            {"enabled": True, "panel_aliases": ["p0"], "escalation_fraction": 0.25}
        )

        # Row 2 alone gets a low base logprob; it must be the one escalated.
        def _varied(data):
            ids = data["input_ids"]
            out = torch.full(ids.shape, -1.0, dtype=torch.float32)
            if ids.shape[0] == 4:
                out[2] = -9.0
            return out

        col._teacher_worker_groups["t1"].score = _varied  # type: ignore[method-assign]
        result, _t = _score(col)
        combined = math.log((math.exp(-9.0) + math.exp(-0.5)) / 2)
        assert result[2].tolist() == pytest.approx([combined] * 3)
        assert result[0].tolist() == pytest.approx([-1.0] * 3)


# ── Wires


class TestWiring:
    def test_the_collector_bursts_after_the_base_pass(self):
        from dockyard_rl.algorithms.async_utils.trajectory_collector import (
            AsyncTrajectoryCollectorImpl,
        )

        source = inspect.getsource(
            AsyncTrajectoryCollectorImpl._compute_teacher_logprobs
        )
        assert "self._burst_teacher_panel(" in source

    def test_the_panel_metrics_reach_the_rollout_metrics(self):
        from dockyard_rl.algorithms.async_utils import trajectory_collector as tc

        source = inspect.getsource(tc)
        assert "rollout_metrics.update(self._teacher_panel_metrics)" in source

    def test_the_worker_group_builder_validates_the_panel(self):
        from dockyard_rl.algorithms.opd import create_teacher_worker_groups

        source = inspect.getsource(create_teacher_worker_groups)
        assert "validate_teacher_panel_config(" in source
        assert "available_aliases=" in source

    def test_the_launcher_refuses_a_panel_without_async(self):
        import importlib.util
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "_run_grpo_swe_f3c", root / "examples/run_grpo_swe.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        config = SimpleNamespace(
            grpo={"async_grpo": {"enabled": False}},
            on_policy_distillation=_opd(teacher_panel=_panel()),
        )
        with pytest.raises(NotImplementedError, match="async_grpo"):
            module._validate_teacher_panel_scope(config)

        config.grpo["async_grpo"]["enabled"] = True
        module._validate_teacher_panel_scope(config)

    def test_the_launcher_check_is_called(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        source = (root / "examples/run_grpo_swe.py").read_text()
        assert "_validate_teacher_panel_scope(config)" in source
        assert source.count("_validate_teacher_panel_scope") == 2
