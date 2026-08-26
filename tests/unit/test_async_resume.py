"""#3599: async resume lost trajectories and skipped the prompts behind them.

Two compounding defects on every async resume.

(a) get_dataloader_state returned self._dataloader.state_dict() -- the
    collector's *live* cursor, already advanced past prompts still generating
    or sitting in the replay buffer. That is what grpo.py wrote to
    train_dataloader.pt, so a resume skipped those prompts entirely; they were
    never trained on and nothing reported it. Checkpointing also runs while
    collection continues (the collector is paused only for validation), so the
    live cursor was racing the save.

(b) The replay buffer was never checkpointed at all. ReplayBuffer exposed
    add/sample/evict/size/clear/get_debug_info/... and no state_dict, and a
    tree-wide grep found no persistence.

They compound: every buffered trajectory was discarded *and* the prompts that
produced it were skipped, so the generation work was unrecoverable in both
directions.

The fix pairs them, because either alone is wrong. With the buffer persisted,
the cursor to save is the position of the last *fully enqueued* batch --
everything at or before it is restored from the buffer and must not be
re-yielded; everything after it is unaccounted for and must be. The batch in
flight at checkpoint time falls in the second group and is correctly
regenerated.
"""

from __future__ import annotations

from typing import Any

import pytest

from dockyard_rl.algorithms.async_utils.replay_buffer import ReplayBufferImpl
from dockyard_rl.algorithms.async_utils.trajectory_collector import (
    AsyncTrajectoryCollectorImpl,
)


def _buffer(max_size: int = 8) -> ReplayBufferImpl:
    return ReplayBufferImpl(max_size=max_size)


def _fill(buf: ReplayBufferImpl, n: int, *, dataplane: bool = False) -> None:
    for i in range(n):
        traj = (
            {"meta": f"m{i}", "driver_carry": f"c{i}", "rollout_metrics": {}}
            if dataplane
            else {"batch": {"tokens": [i, i + 1]}, "rollout_metrics": {"n": i}}
        )
        buf.add(traj, weight_version=i, target_weight_version=i + 2)


class TestBufferRoundTrip:
    def test_a_restored_buffer_matches_the_saved_one(self):
        src = _buffer()
        _fill(src, 3)
        state = src.state_dict()

        dst = _buffer()
        assert dst.load_state_dict(state) is True
        assert dst.size() == 3
        assert dst.trajectories == src.trajectories
        assert dst.trajectory_versions == src.trajectory_versions
        assert dst.target_weight_versions == src.target_weight_versions

    def test_the_generation_watermark_survives(self):
        """The collector reads it to avoid regenerating covered targets."""
        src = _buffer()
        _fill(src, 3)
        dst = _buffer()
        dst.load_state_dict(src.state_dict())
        assert (
            dst.get_last_target_weight_already_generated()
            == src.get_last_target_weight_already_generated()
        )

    def test_a_restored_buffer_is_samplable(self):
        """Restoring fields without restoring behaviour would not help.

        _fill sets target_weight_version = i + 2, and sample() matches on that
        target, so the step asked for has to be one the buffer actually holds.
        """
        src = _buffer()
        _fill(src, 4)
        dst = _buffer()
        dst.load_state_dict(src.state_dict())
        assert dst.sample(
            num_prompt_groups=1, current_weight_version=2, max_age_steps=100
        ) is not None

    def test_an_empty_buffer_round_trips(self):
        dst = _buffer()
        assert dst.load_state_dict(_buffer().state_dict()) is True
        assert dst.size() == 0

    def test_the_snapshot_does_not_alias_live_state(self):
        """The buffer keeps mutating after the checkpoint is taken."""
        src = _buffer()
        _fill(src, 2)
        state = src.state_dict()
        _fill(src, 2)
        assert len(state["trajectories"]) == 2


class TestDataPlaneEntriesAreRefused:
    """Saving them would restore a buffer of dangling handles."""

    def test_state_dict_marks_them_unpersisted(self):
        buf = _buffer()
        _fill(buf, 2, dataplane=True)
        state = buf.state_dict()
        assert state["persisted"] is False
        assert state["discarded_groups"] == 2
        assert "trajectories" not in state

    def test_loading_such_a_checkpoint_starts_empty(self):
        buf = _buffer()
        _fill(buf, 2, dataplane=True)
        dst = _buffer()
        assert dst.load_state_dict(buf.state_dict()) is False
        assert dst.size() == 0

    def test_a_legacy_buffer_is_still_persisted(self):
        """Discrimination: the refusal must not swallow the normal path."""
        buf = _buffer()
        _fill(buf, 2, dataplane=False)
        assert buf.state_dict()["persisted"] is True


class TestRestoreRefusesToTruncate:
    def test_a_smaller_max_size_starts_empty_rather_than_dropping_groups(self):
        src = _buffer(max_size=8)
        _fill(src, 6)
        dst = _buffer(max_size=4)
        assert dst.load_state_dict(src.state_dict()) is False
        assert dst.size() == 0

    def test_an_equal_max_size_restores(self):
        src = _buffer(max_size=4)
        _fill(src, 4)
        dst = _buffer(max_size=4)
        assert dst.load_state_dict(src.state_dict()) is True
        assert dst.size() == 4


class _FakeDataLoader:
    """A cursor that advances one position per batch, like StatefulDataLoader."""

    def __init__(self) -> None:
        self.position = 0

    def state_dict(self) -> dict[str, Any]:
        return {"position": self.position}


def _collector_with_loader() -> tuple[Any, _FakeDataLoader]:
    c = object.__new__(AsyncTrajectoryCollectorImpl)
    loader = _FakeDataLoader()
    c._dataloader = loader
    c._enqueued_dataloader_state = None
    return c, loader


class TestCheckpointedCursorIsTheFrontier:
    def test_nothing_enqueued_yet_checkpoints_nothing(self):
        """A resume then restarts the epoch, having lost nothing."""
        c, _ = _collector_with_loader()
        assert c.get_dataloader_state() is None

    def test_the_cursor_reflects_the_last_enqueued_batch(self):
        c, loader = _collector_with_loader()
        loader.position = 1
        c._enqueued_dataloader_state = loader.state_dict()
        loader.position = 2  # a batch was pulled and is now in flight
        assert c.get_dataloader_state() == {"position": 1}

    def test_an_in_flight_batch_does_not_advance_the_cursor(self):
        """This is defect (a): the live cursor had already moved past it."""
        c, loader = _collector_with_loader()
        loader.position = 3
        c._enqueued_dataloader_state = loader.state_dict()

        for _ in range(5):  # five more batches pulled, none enqueued
            loader.position += 1

        assert c.get_dataloader_state() == {"position": 3}, (
            "the checkpointed cursor followed the live one, so a resume would "
            "skip every prompt pulled but not yet enqueued"
        )

    def test_the_cursor_only_moves_forward_as_batches_complete(self):
        c, loader = _collector_with_loader()
        seen = []
        for _ in range(3):
            loader.position += 1
            c._enqueued_dataloader_state = loader.state_dict()
            seen.append(c.get_dataloader_state()["position"])
        assert seen == [1, 2, 3]


class TestTheTwoHalvesAgree:
    """The pair is the point: either alone still loses work."""

    def test_buffered_prompts_are_restored_and_not_re_yielded(self):
        c, loader = _collector_with_loader()
        buf = _buffer()

        # Three batches generated and enqueued; cursor advances with each.
        for i in range(3):
            buf.add(
                {"batch": {"tokens": [i]}, "rollout_metrics": {}},
                weight_version=i,
                target_weight_version=i + 1,
            )
            loader.position += 1
            c._enqueued_dataloader_state = loader.state_dict()

        # A fourth is pulled and is still generating when the checkpoint lands.
        loader.position += 1

        cursor = c.get_dataloader_state()
        state = buf.state_dict()

        # Resume.
        restored = _buffer()
        restored.load_state_dict(state)

        assert restored.size() == 3, "the three enqueued groups must come back"
        assert cursor == {"position": 3}, (
            "the cursor must sit after the three restored batches and before "
            "the in-flight fourth, so the fourth is regenerated and the first "
            "three are not trained on twice"
        )
