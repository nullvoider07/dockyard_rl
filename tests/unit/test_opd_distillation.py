"""CPU tests for on-policy distillation (OPD/MOPD) core: config helpers, teacher
routing, the prev_logprobs guard, the teacher-config builder, and the
OPDAdvantageEstimator math.

The teacher worker group + async-GRPO orchestration are GPU/cluster-deferred.
"""

import threading
from types import SimpleNamespace

import pytest
from typing import Any

import torch

from dockyard_rl.algorithms import opd
from dockyard_rl.algorithms.advantage_estimator import OPDAdvantageEstimator


# -- config helpers -----------------------------------------------------------

def test_is_opd_enabled_dict_basemodel_and_missing():
    assert opd.is_opd_enabled({"on_policy_distillation": {"enabled": True}}) is True
    assert opd.is_opd_enabled({"on_policy_distillation": {"enabled": False}}) is False
    # missing field (non-OPD recipe)
    assert opd.is_opd_enabled(SimpleNamespace()) is False
    # BaseModel form
    cfg = SimpleNamespace(
        on_policy_distillation=opd.OnPolicyDistillationConfig(enabled=True)
    )
    assert opd.is_opd_enabled(cfg) is True


def test_is_non_colocated_teachers_enabled():
    cfg = {
        "on_policy_distillation": {
            "enabled": True,
            "non_colocated_teachers": {"enabled": True},
        }
    }
    assert opd.is_non_colocated_teachers_enabled(cfg) is True
    # OPD off short-circuits to False even if non_colocated is set.
    cfg["on_policy_distillation"]["enabled"] = False
    assert opd.is_non_colocated_teachers_enabled(cfg) is False


def _master_cfg(force_on_policy_ratio, seq_logprob_error_threshold, opd_enabled=True):
    return SimpleNamespace(
        loss_fn=SimpleNamespace(force_on_policy_ratio=force_on_policy_ratio),
        grpo={"seq_logprob_error_threshold": seq_logprob_error_threshold},
        on_policy_distillation={"enabled": opd_enabled},
    )


def test_assert_prev_logprobs_available_raises_when_zeroed():
    # force_on_policy_ratio=True with no threshold zeros prev_logprobs -> raise.
    bad = _master_cfg(force_on_policy_ratio=True, seq_logprob_error_threshold=None)
    with pytest.raises(ValueError, match="real prev_logprobs"):
        opd.assert_prev_logprobs_available(bad)


def test_assert_prev_logprobs_available_ok_with_threshold_or_no_force():
    # A threshold keeps the student logprob pass alive.
    opd.assert_prev_logprobs_available(
        _master_cfg(force_on_policy_ratio=True, seq_logprob_error_threshold=0.1)
    )
    # No forcing -> prev_logprobs computed normally.
    opd.assert_prev_logprobs_available(
        _master_cfg(force_on_policy_ratio=False, seq_logprob_error_threshold=None)
    )
    # OPD disabled -> guard is inert even in the zeroing config.
    opd.assert_prev_logprobs_available(
        _master_cfg(force_on_policy_ratio=True, seq_logprob_error_threshold=None, opd_enabled=False)
    )


# -- teacher routing ----------------------------------------------------------

def test_resolve_reference_aliases_mapped_and_default_fallback():
    mapping = {"agentA": "modelA", "fallback": "modelF"}
    refs = [{"name": "agentA"}, {"name": "agentB"}]
    aliases = opd.resolve_reference_aliases(
        refs, mapping, default_teacher_alias="fallback"
    )
    assert aliases == ["agentA", "fallback"]


def test_resolve_reference_aliases_strict_raises():
    mapping = {"agentA": "modelA"}
    with pytest.raises(ValueError, match="No teacher model mapping"):
        opd.resolve_reference_aliases(
            [{"name": "agentB"}], mapping, strict_agent_name_match=True
        )


def test_resolve_reference_aliases_no_default_raises():
    with pytest.raises(ValueError, match="no default_teacher_alias"):
        opd.resolve_reference_aliases([{"name": "x"}], {"a": "m"})


def test_get_teacher_routing_metrics_compression():
    # Two aliases share one underlying model -> compression 0.5.
    mapping = {"a": "shared", "b": "shared"}
    m = opd.get_teacher_routing_metrics(["a", "b"], mapping)
    assert m["on_policy_distillation/teacher_alias_unique"] == 2.0
    assert m["on_policy_distillation/teacher_model_unique"] == 1.0
    assert m["on_policy_distillation/teacher_alias_to_model_compression"] == 0.5


def test_get_teacher_routing_metrics_missing_alias_raises():
    with pytest.raises(KeyError):
        opd.get_teacher_routing_metrics(["ghost"], {"a": "m"})


# -- teacher config builder (dedup + overrides + extra->dtensor) --------------

def test_create_teacher_configs_dedup_and_overrides():
    from dockyard_rl.models.policy.teacher_worker_group import (
        create_teacher_configs_from_opd_config,
    )

    opd_cfg = {
        "teacher_model_by_agent_name": {"a": "ckpt1", "b": "ckpt1", "c": "ckpt2"},
        "deduplicate_shared_teacher_checkpoints": True,
        "non_colocated_teachers": {
            "default_teacher_cfg": {"tensor_parallel_size": 2, "num_nodes": 1},
            "teacher_overrides": {"c": {"tensor_parallel_size": 4, "unknown_knob": 7}},
        },
    }
    cfgs = create_teacher_configs_from_opd_config(opd_cfg)
    # a and b share ckpt1 -> deduped to one; c is separate -> 2 configs.
    assert len(cfgs) == 2
    by_alias = {c.alias: c for c in cfgs}
    assert by_alias["a"].tensor_parallel_size == 2
    assert by_alias["c"].tensor_parallel_size == 4
    # Unknown top-level key folds into dtensor_cfg_overrides.
    assert by_alias["c"].dtensor_cfg_overrides.get("unknown_knob") == 7


# -- OPDAdvantageEstimator ----------------------------------------------------

def test_opd_advantage_is_teacher_minus_student_masked():
    est = OPDAdvantageEstimator({}, {})
    teacher = torch.tensor([[1.0, 2.0, 3.0]])
    student = torch.tensor([[0.5, 1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    adv = est.compute_advantage(
        prompt_ids=torch.tensor([0]),
        rewards=torch.tensor([0.0]),
        mask=mask,
        logprobs_policy=student,
        teacher_logprobs=teacher,
    )
    expected = torch.tensor([[0.5, 1.0, 0.0]])  # (teacher-student) * mask
    assert torch.allclose(adv, expected)
    # stop-gradient: advantage carries no grad even from grad-requiring inputs.
    t = teacher.clone().requires_grad_(True)
    adv2 = est.compute_advantage(
        prompt_ids=None, rewards=None, mask=mask,
        logprobs_policy=student, teacher_logprobs=t,
    )
    assert not adv2.requires_grad


def test_opd_advantage_requires_both_logprobs():
    est = OPDAdvantageEstimator({}, {})
    mask = torch.ones(1, 3)
    with pytest.raises(ValueError, match="teacher_logprobs"):
        est.compute_advantage(None, None, mask, logprobs_policy=torch.zeros(1, 3))
    with pytest.raises(ValueError, match="logprobs_policy"):
        est.compute_advantage(None, None, mask, teacher_logprobs=torch.zeros(1, 3))


def test_opd_advantage_metrics_populated():
    est = OPDAdvantageEstimator({}, {})
    teacher = torch.tensor([[1.0, 2.0]])
    student = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0]])
    est.compute_advantage(None, None, mask, logprobs_policy=student, teacher_logprobs=teacher)
    m = est.last_metrics
    assert m["on_policy_distillation/teacher_student_logprob_gap_mean"] == pytest.approx(1.5)
    assert "on_policy_distillation/adv_mean" in m
    assert "on_policy_distillation/adv_std" in m


# -- the prev_logprobs guard accepts every config shape the module claims -----

def _opd_on() -> dict:
    return {"enabled": True, "default_teacher_alias": "t1"}


def test_prev_logprobs_guard_accepts_a_plain_dict_config():
    """`_opd_cfg` documents dict configs and `is_opd_enabled` honours them.

    The guard did not: it read master_config.loss_fn directly, so a dict raised
    AttributeError from inside the one function whose job is to report a
    different misconfiguration clearly.
    """
    from dockyard_rl.algorithms.opd import assert_prev_logprobs_available

    cfg = {
        "on_policy_distillation": _opd_on(),
        "loss_fn": {"force_on_policy_ratio": False},
        "grpo": {"seq_logprob_error_threshold": None},
    }

    assert_prev_logprobs_available(cfg)  # must not raise


def test_prev_logprobs_guard_still_refuses_a_bad_dict_config():
    """Tolerating the shape must not stop it catching the thing it exists for."""
    from dockyard_rl.algorithms.opd import assert_prev_logprobs_available

    cfg = {
        "on_policy_distillation": _opd_on(),
        "loss_fn": {"force_on_policy_ratio": True},
        "grpo": {"seq_logprob_error_threshold": None},
    }

    with pytest.raises(ValueError, match="requires real prev_logprobs"):
        assert_prev_logprobs_available(cfg)


def test_prev_logprobs_guard_accepts_an_object_config():
    """The MasterConfig shape the real call site passes."""
    from types import SimpleNamespace

    from dockyard_rl.algorithms.opd import assert_prev_logprobs_available

    master = SimpleNamespace(
        on_policy_distillation=_opd_on(),
        loss_fn=SimpleNamespace(force_on_policy_ratio=True),
        grpo={"seq_logprob_error_threshold": 1.5},
    )

    assert_prev_logprobs_available(master)  # threshold set -> not skipping


def test_a_missing_section_reads_as_not_skipping():
    """Absent force_on_policy_ratio means the logprob pass runs, so nothing to refuse."""
    from dockyard_rl.algorithms.opd import assert_prev_logprobs_available

    assert_prev_logprobs_available({"on_policy_distillation": _opd_on()})


def test_the_guard_is_inert_when_opd_is_off():
    """A non-OPD recipe must not be touched by any of this."""
    from dockyard_rl.algorithms.opd import assert_prev_logprobs_available

    assert_prev_logprobs_available(
        {"loss_fn": {"force_on_policy_ratio": True}, "grpo": {}}
    )


# -- collector teacher scoring (routing + DP padding + stitching) -------------

class _FakeScorer:
    """A TeacherScorer that runs nowhere.

    This is the point of the protocol: the collector's routing, row-slicing and
    stitching are exercised with no cluster, no sharding and no transport. It
    pads nothing -- batch-axis padding is an implementation's own business now,
    so a scorer that needs none has none, and the collector must not assume
    otherwise.
    """

    requires_prepadded_input = False
    sequence_pad_multiple = 1

    def __init__(self, alias="t1", logprob_fn=None, seen=None):
        self.alias = alias
        self._logprob_fn = logprob_fn or (
            lambda ids: torch.zeros(ids.shape, dtype=torch.float32)
        )
        self._seen = seen
        self.calls = 0

    def score(self, data):
        self.calls += 1
        if self._seen is not None:
            self._seen[self.alias] = data
        return self._logprob_fn(data["input_ids"])

    def shutdown(self):
        return True


def _make_collector_with_teacher(teacher_logprob_fn, dp_size=1):
    """Bypass-construct an AsyncTrajectoryCollectorImpl with one fake scorer."""
    from dockyard_rl.algorithms.async_utils import trajectory_collector as tc

    col = tc.AsyncTrajectoryCollectorImpl.__new__(tc.AsyncTrajectoryCollectorImpl)
    teacher = _FakeScorer(logprob_fn=teacher_logprob_fn)
    col._teacher_worker_groups = {"t1": teacher}
    col._alias_to_group_alias = {"t1": "t1"}
    col._on_policy_distillation_cfg = {
        "teacher_model_by_agent_name": {"t1": "m1"},
        "default_teacher_alias": "t1",
    }
    return col, teacher


def test_collector_teacher_logprobs_routes_and_stitches():
    col, _ = _make_collector_with_teacher(
        lambda ids: torch.arange(ids.numel(), dtype=torch.float32).reshape(ids.shape)
    )
    input_ids = torch.zeros(3, 4, dtype=torch.long)
    agent_refs = [{"name": "t1"}, {"name": "t1"}, {"name": "t1"}]
    result, _t = col._compute_teacher_logprobs(input_ids, agent_refs)
    assert result.shape == (3, 4)
    assert torch.allclose(result, torch.arange(12, dtype=torch.float32).reshape(3, 4))


def test_the_collector_does_not_pad_the_batch_axis_itself():
    """It hands over the rows it routed; padding is the scorer's own business.

    A collector that padded here would be padding for a sharding a remote or
    API scorer does not have.
    """
    col, seen = _collector_capturing_sub_data()

    col._compute_teacher_logprobs(
        torch.zeros(3, 5, dtype=torch.long), [{"name": "t1"}] * 3
    )

    assert seen["t1"]["input_ids"].shape[0] == 3


def _collector_capturing_sub_data(dp_size=1, aliases=("t1",)):
    """Collector whose fake scorers record the BatchedDataDict they were given."""
    from dockyard_rl.algorithms.async_utils import trajectory_collector as tc

    col = tc.AsyncTrajectoryCollectorImpl.__new__(tc.AsyncTrajectoryCollectorImpl)
    seen: dict[str, Any] = {}
    groups = {alias: _FakeScorer(alias=alias, seen=seen) for alias in aliases}

    col._teacher_worker_groups = groups
    col._alias_to_group_alias = {a: a for a in aliases}
    col._on_policy_distillation_cfg = {
        "teacher_model_by_agent_name": {a: f"m_{a}" for a in aliases},
        "default_teacher_alias": aliases[0],
    }
    return col, seen


# -- #3430b: multimodal payload reaches the non-colocated teacher -------------

def test_multimodal_columns_reach_the_teacher():
    """The defect: the teacher was called with input_ids and nothing else."""
    col, seen = _collector_capturing_sub_data()
    pixels = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    col._compute_teacher_logprobs(
        torch.zeros(3, 4, dtype=torch.long),
        [{"name": "t1"}] * 3,
        multimodal_data={"pixel_values": pixels},
    )

    assert "pixel_values" in seen["t1"]
    assert torch.equal(seen["t1"]["pixel_values"], pixels)


def test_each_teacher_receives_only_its_own_rows():
    """Routing splits the batch; the images have to split the same way."""
    col, seen = _collector_capturing_sub_data(aliases=("t1", "t2"))
    # Rows 0 and 2 -> t1, row 1 -> t2.
    marks = ["img0", "img1", "img2"]

    col._compute_teacher_logprobs(
        torch.zeros(3, 4, dtype=torch.long),
        [{"name": "t1"}, {"name": "t2"}, {"name": "t1"}],
        multimodal_data={"image_sizes": marks},
    )

    assert seen["t1"]["image_sizes"] == ["img0", "img2"]
    assert seen["t2"]["image_sizes"] == ["img1"]


class TestTheWorkerGroupPadsItsOwnBatchAxis:
    """The padding half of #3430b, now tested where the padding lives.

    A DTensor teacher needs its batch to divide by its data-parallel size, so
    `score` repeats the last row up to a multiple and trims the result back. The
    caller never sees it. Multimodal columns must repeat the *same* row: padded
    with anything else -- or not padded -- a column arrives misaligned and the
    teacher scores row i's tokens against row j's image.
    """

    def _group(self, dp_size, seen):
        from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
        from dockyard_rl.models.policy.teacher_worker_group import TeacherWorkerGroup

        group = TeacherWorkerGroup.__new__(TeacherWorkerGroup)
        group._dp_size = dp_size
        group._score_lock = threading.Lock()

        def _get_logprobs(data, micro_batch_size=None):
            seen.append(data)
            ids = data["input_ids"]
            return BatchedDataDict(
                {
                    "reference_logprobs": torch.arange(
                        ids.shape[0], dtype=torch.float32
                    )
                    .unsqueeze(-1)
                    .expand(ids.shape)
                    .clone()
                }
            )

        group.get_logprobs = _get_logprobs  # type: ignore[method-assign]
        return group

    def test_the_batch_is_padded_up_and_the_result_trimmed_back(self):
        from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

        seen: list[Any] = []
        group = self._group(2, seen)

        out = group.score(
            BatchedDataDict({"input_ids": torch.zeros(3, 5, dtype=torch.long)})
        )

        assert seen[0]["input_ids"].shape[0] == 4
        assert out.shape == (3, 5)

    def test_the_pad_repeats_the_same_row_in_every_column(self):
        from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

        seen: list[Any] = []
        group = self._group(2, seen)

        group.score(
            BatchedDataDict(
                {
                    "input_ids": torch.zeros(3, 5, dtype=torch.long),
                    "image_sizes": ["a", "b", "c"],
                }
            )
        )

        assert seen[0]["input_ids"].shape[0] == 4
        assert seen[0]["image_sizes"] == ["a", "b", "c", "c"]

    def test_an_already_divisible_batch_is_passed_through_untouched(self):
        """Inert on the common path: no copy, no new dict."""
        from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

        seen: list[Any] = []
        group = self._group(2, seen)
        data = BatchedDataDict({"input_ids": torch.zeros(4, 5, dtype=torch.long)})

        group.score(data)

        assert seen[0] is data

    def test_the_pad_rows_never_reach_the_caller(self):
        """A leaked pad row shifts every later row onto the wrong sample."""
        from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

        seen: list[Any] = []
        group = self._group(4, seen)

        out = group.score(
            BatchedDataDict({"input_ids": torch.zeros(3, 2, dtype=torch.long)})
        )

        assert seen[0]["input_ids"].shape[0] == 4
        torch.testing.assert_close(out[:, 0], torch.tensor([0.0, 1.0, 2.0]))


def test_a_packed_tensor_column_is_sliced_by_row():
    """PackedTensor is sample-indexed and exposes .slice; use it, don't stack."""
    from dockyard_rl.data.multimodal_utils import PackedTensor

    col, seen = _collector_capturing_sub_data(aliases=("t1", "t2"))
    packed = PackedTensor(
        [torch.full((1, 2), float(i)) for i in range(3)], dim_to_pack=0
    )

    col._compute_teacher_logprobs(
        torch.zeros(3, 4, dtype=torch.long),
        [{"name": "t1"}, {"name": "t2"}, {"name": "t1"}],
        multimodal_data={"pixel_values": packed},
    )

    got = seen["t1"]["pixel_values"]
    assert isinstance(got, PackedTensor)
    assert all(t is not None for t in got.tensors)
    assert [float(t[0, 0]) for t in got.tensors if t is not None] == [0.0, 2.0]


def test_a_text_only_batch_is_unchanged():
    """Inert default: no multimodal keys means the sub-batch is what it was."""
    col, seen = _collector_capturing_sub_data()

    col._compute_teacher_logprobs(
        torch.zeros(2, 3, dtype=torch.long), [{"name": "t1"}] * 2
    )

    assert set(seen["t1"].keys()) == {"input_ids"}


def test_a_column_that_cannot_be_row_selected_is_refused():
    """Forwarding it whole is the misalignment this fix removes."""
    col, _ = _collector_capturing_sub_data()

    with pytest.raises(TypeError, match="row-selected"):
        col._compute_teacher_logprobs(
            torch.zeros(2, 3, dtype=torch.long),
            [{"name": "t1"}] * 2,
            multimodal_data={"pixel_values": object()},
        )


def test_the_scorer_passes_the_multimodal_columns():
    """The wire: _score_teacher_logprobs must read them off the flat batch."""
    import inspect

    from dockyard_rl.algorithms.async_utils import trajectory_collector as tc

    source = inspect.getsource(
        tc.AsyncTrajectoryCollectorImpl._score_teacher_logprobs
    )

    assert "get_multimodal_dict(as_tensors=False)" in source
    assert "multimodal_data=" in source


def test_collector_teacher_logprobs_default_routing_for_unknown_agent():
    col, _ = _make_collector_with_teacher(
        lambda ids: torch.full(ids.shape, 7.0, dtype=torch.float32)
    )
    # Unknown agent name -> falls back to default_teacher_alias 't1'.
    result, _t = col._compute_teacher_logprobs(
        torch.zeros(2, 3, dtype=torch.long), [{"name": "ghost"}, {"name": "ghost"}]
    )
    assert torch.allclose(result, torch.full((2, 3), 7.0))
