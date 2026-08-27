"""E2: a teacher is whatever can score trajectories, not a local worker group.

OPD teachers are the one class of model dockyard never refits -- loaded frozen
and only ever asked for logprobs -- so they are the one fleet whose
implementation is interchangeable. The code did not read that way. The OPD path
reached into the teacher object for four things only a locally sharded worker
group has: its data-parallel axis size (to pad a sub-batch up to a multiple of
it), its packing mode, its sequence pad multiple, and a lock serializing
collectives across its workers.

`TeacherScorer` is the boundary. The batch-axis pad and the serialization move
behind it; the sequence-axis pre-pad deliberately stays in front, because one
flattened batch is scored by every teacher and cannot be padded per teacher.

The tests below are the claim made concrete: a scorer with no cluster, no
sharding and no transport drives the whole routing/slicing/stitching path. That
is what E2 buys, and it is checkable with no hardware -- which is the entire
reason E2 is reachable here and E1 is not.
"""

from __future__ import annotations

import inspect
import pathlib
from typing import Any

import pytest
import torch

from dockyard_rl.algorithms.opd import teacher_seq_pad_multiple
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.models.policy.teacher_scorer import TeacherScorer, select_rows


class _RemoteScorer:
    """A teacher that runs somewhere else entirely.

    No DP axis, no packing mode, no worker group -- the shape an API-served or
    remote-endpoint teacher actually has. It exists to prove the consuming code
    never asks for those, so it deliberately defines nothing beyond the
    protocol.
    """

    requires_prepadded_input = False
    sequence_pad_multiple = 1

    def __init__(self, alias: str = "remote") -> None:
        self.alias = alias
        self.batches: list[Any] = []

    def score(self, data: Any) -> torch.Tensor:
        self.batches.append(data)
        ids = data["input_ids"]
        # One row out per row in, and identifiable per row.
        return (
            torch.arange(ids.shape[0], dtype=torch.float32)
            .unsqueeze(-1)
            .expand(ids.shape)
            .clone()
        )

    def shutdown(self) -> bool:
        return True


class _LocalishScorer(_RemoteScorer):
    """A scorer that wants the caller to pre-pad the sequence axis."""

    requires_prepadded_input = True

    def __init__(self, alias: str = "local", pad_multiple: int = 8) -> None:
        super().__init__(alias)
        self.sequence_pad_multiple = pad_multiple  # type: ignore[misc]


class TestTheProtocolAdmitsBothKinds:
    def test_a_remote_scorer_satisfies_it(self):
        assert isinstance(_RemoteScorer(), TeacherScorer)

    def test_an_object_missing_score_does_not(self):
        class NotAScorer:
            alias = "x"
            requires_prepadded_input = False
            sequence_pad_multiple = 1

            def shutdown(self) -> bool:
                return True

        assert not isinstance(NotAScorer(), TeacherScorer)

    def test_the_worker_group_declares_the_whole_surface(self):
        """The local implementation must not drift away from the contract."""
        from dockyard_rl.models.policy.teacher_worker_group import TeacherWorkerGroup

        for member in ("requires_prepadded_input", "sequence_pad_multiple", "score",
                       "shutdown"):
            assert hasattr(TeacherWorkerGroup, member), (
                f"TeacherWorkerGroup no longer provides {member!r}; the OPD path "
                "reads it through TeacherScorer"
            )

    def test_score_is_the_only_scoring_entry_point_in_the_protocol(self):
        """`get_logprobs` is the local implementation's own call, not the contract.

        A remote teacher has no `get_logprobs`; if the protocol grew one, every
        consumer would be free to use it and the boundary would be back where it
        started.
        """
        assert not hasattr(TeacherScorer, "get_logprobs")


class TestTheSequencePadNegotiation:
    """The one pad that stays in front of the boundary, and why."""

    def test_no_teachers_needs_no_prepad(self):
        assert teacher_seq_pad_multiple({}, 16) == 1

    def test_scorers_that_pad_themselves_need_no_prepad(self):
        scorers = {"a": _RemoteScorer("a"), "b": _RemoteScorer("b")}
        assert teacher_seq_pad_multiple(scorers, 16) == 1

    def test_scorers_that_want_it_get_the_policy_divisor(self):
        scorers = {"a": _LocalishScorer("a", 8), "b": _LocalishScorer("b", 4)}
        assert teacher_seq_pad_multiple(scorers, 16) == 16

    def test_a_divisor_that_does_not_satisfy_every_scorer_is_refused(self):
        scorers = {"a": _LocalishScorer("a", 8), "b": _LocalishScorer("b", 5)}
        with pytest.raises(ValueError, match="must be a multiple of teacher 'b'"):
            teacher_seq_pad_multiple(scorers, 16)

    def test_mixed_prepad_expectations_are_refused(self):
        """One flattened batch cannot be padded both ways."""
        scorers = {"a": _RemoteScorer("a"), "b": _LocalishScorer("b", 8)}
        with pytest.raises(ValueError, match="agree on sequence pre-padding"):
            teacher_seq_pad_multiple(scorers, 16)

    def test_it_reads_the_protocol_and_not_a_worker_group_attribute(self):
        """Cut the rename and a remote scorer raises AttributeError here."""
        source = inspect.getsource(teacher_seq_pad_multiple)
        assert "use_sequence_packing" not in source
        assert "sequence_length_pad_multiple" not in source


class TestSelectRows:
    def test_a_tensor_column_is_indexed(self):
        value = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        torch.testing.assert_close(
            select_rows("pixel_values", value, [2, 0]), value[[2, 0]]
        )

    def test_a_list_column_is_indexed(self):
        assert select_rows("image_sizes", ["a", "b", "c"], [1, 1]) == ["b", "b"]

    def test_a_packed_tensor_goes_through_its_own_slice(self):
        from dockyard_rl.data.multimodal_utils import PackedTensor

        packed = PackedTensor(
            [torch.full((1, 2), float(i)) for i in range(3)], dim_to_pack=0
        )
        got = select_rows("pixel_values", packed, [0, 2])

        assert isinstance(got, PackedTensor)
        tensors = [t for t in got.tensors if t is not None]
        assert [float(t[0, 0]) for t in tensors] == [0.0, 2.0]

    def test_repeating_an_index_is_how_a_pad_row_is_made(self):
        assert select_rows("image_sizes", ["a", "b", "c"], [0, 1, 2, 2]) == [
            "a",
            "b",
            "c",
            "c",
        ]

    def test_a_column_that_cannot_be_row_selected_is_refused(self):
        """Forwarding it whole would misalign it with the rows being scored."""
        with pytest.raises(TypeError, match="cannot be"):
            select_rows("weird", object(), [0])


class TestARemoteTeacherDrivesTheWholeOPDPath:
    """The end-to-end claim: no cluster, no sharding, no transport."""

    def _collector(self, scorers: dict[str, Any]):
        from dockyard_rl.algorithms.async_utils import trajectory_collector as tc

        col = tc.AsyncTrajectoryCollectorImpl.__new__(tc.AsyncTrajectoryCollectorImpl)
        col._teacher_worker_groups = scorers
        col._alias_to_group_alias = {a: a for a in scorers}
        col._on_policy_distillation_cfg = {
            "teacher_model_by_agent_name": {a: f"m_{a}" for a in scorers},
            "default_teacher_alias": next(iter(scorers)),
        }
        return col

    def test_logprobs_are_stitched_back_onto_the_right_rows(self):
        code, reasoning = _RemoteScorer("code"), _RemoteScorer("reasoning")
        col = self._collector({"code": code, "reasoning": reasoning})

        result, _ = col._compute_teacher_logprobs(
            torch.zeros(4, 3, dtype=torch.long),
            [
                {"name": "code"},
                {"name": "reasoning"},
                {"name": "code"},
                {"name": "reasoning"},
            ],
        )

        # Each scorer numbers its own sub-batch 0..n-1; stitching puts them back
        # on the flattened batch's rows.
        assert result.shape == (4, 3)
        torch.testing.assert_close(
            result[:, 0], torch.tensor([0.0, 0.0, 1.0, 1.0])
        )

    def test_each_scorer_sees_only_its_own_rows(self):
        code, reasoning = _RemoteScorer("code"), _RemoteScorer("reasoning")
        col = self._collector({"code": code, "reasoning": reasoning})

        col._compute_teacher_logprobs(
            torch.zeros(3, 3, dtype=torch.long),
            [{"name": "code"}, {"name": "reasoning"}, {"name": "code"}],
            multimodal_data={"image_sizes": ["i0", "i1", "i2"]},
        )

        assert code.batches[0]["image_sizes"] == ["i0", "i2"]
        assert reasoning.batches[0]["image_sizes"] == ["i1"]

    def test_the_rows_handed_over_are_exactly_the_rows_routed(self):
        """No batch-axis padding in front of the boundary."""
        remote = _RemoteScorer()
        col = self._collector({"remote": remote})

        col._compute_teacher_logprobs(
            torch.zeros(3, 3, dtype=torch.long), [{"name": "remote"}] * 3
        )

        assert remote.batches[0]["input_ids"].shape[0] == 3

    def test_input_lengths_ride_along_when_present(self):
        remote = _RemoteScorer()
        col = self._collector({"remote": remote})

        col._compute_teacher_logprobs(
            torch.zeros(3, 3, dtype=torch.long),
            [{"name": "remote"}] * 3,
            input_lengths=torch.tensor([3, 2, 1]),
        )

        torch.testing.assert_close(
            remote.batches[0]["input_lengths"], torch.tensor([3, 2, 1])
        )


class TestTheCollectorHoldsNoLocalTeacherState:
    """What the boundary removed, pinned so it cannot come back.

    Each of these was a local-worker-group detail living in the collector. A
    remote scorer has no answer for any of them.
    """

    def _collector_source(self) -> str:
        from dockyard_rl.algorithms.async_utils import trajectory_collector

        return pathlib.Path(trajectory_collector.__file__).read_text()

    @pytest.mark.parametrize(
        "leak",
        [
            "sharding_annotations",
            "_teacher_locks",
            "use_sequence_packing",
            "sequence_length_pad_multiple",
        ],
    )
    def test_the_collector_no_longer_reads_it(self, leak):
        assert leak not in self._collector_source(), (
            f"{leak!r} is back in the trajectory collector; that is a property "
            "of a local sharded worker group, so a remote scorer cannot answer it"
        )

    def test_the_collector_calls_score_rather_than_get_logprobs(self):
        source = self._collector_source()
        assert "scorer.score(" in source
        assert ".get_logprobs(" not in source
