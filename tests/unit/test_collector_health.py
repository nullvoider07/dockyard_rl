"""#2368: rollout failures were swallowed forever with no counter and no signal.

AsyncTrajectoryCollector caught every rollout exception, printed it, slept
0.5s and continued -- unbounded, with no counter and nothing the trainer could
read. The trainer meanwhile waits for the replay buffer to fill in its own
`while True` (grpo.py:2557). A wedged generation worker therefore hung both
sides indefinitely: no error, no exit, just "Wait iteration N" forever.

Two halves are tested here. The collector must count failures, distinguish a
streak from isolated errors, and stop rather than spin; the trainer must read
that and abort with a message naming the cause.

The collector is exercised through a bare instance rather than a Ray actor:
the accounting is plain synchronous state, and its serialisation guarantee
comes from the actor event loop, which a unit test cannot reproduce anyway.
"""

from __future__ import annotations

from typing import Any

import pytest

from dockyard_rl.algorithms.grpo import _raise_if_collector_unhealthy
from dockyard_rl.algorithms.async_utils.trajectory_collector import (
    AsyncTrajectoryCollectorImpl,
)


def _collector(max_failures: int = 3) -> Any:
    """A collector with only the failure-accounting state initialised."""
    c = object.__new__(AsyncTrajectoryCollectorImpl)
    c._max_generation_failures = max_failures
    c._consecutive_failures = 0
    c._total_failures = 0
    c._failures_by_stage = {}
    c._failures_by_type = {}
    c._last_error = None
    return c


class TestFailuresAreCounted:
    def test_a_failure_increments_both_counters(self):
        c = _collector()
        c._record_failure("generation", RuntimeError("boom"))
        assert c._consecutive_failures == 1
        assert c._total_failures == 1

    def test_failures_are_broken_down_by_stage(self):
        """Generation timing out and teacher scoring OOMing need different fixes."""
        c = _collector(max_failures=10)
        c._record_failure("generation", RuntimeError("boom"))
        c._record_failure("teacher_scoring", ValueError("bad"))
        c._record_failure("generation", RuntimeError("boom"))

        health = c.check_health()
        assert health["failures_by_stage"] == {"generation": 2, "teacher_scoring": 1}
        assert health["failures_by_type"] == {"RuntimeError": 2, "ValueError": 1}

    def test_the_last_error_is_reported(self):
        c = _collector()
        c._record_failure("generation", RuntimeError("cuda oom"))
        assert "cuda oom" in c.check_health()["last_error"]
        assert "RuntimeError" in c.check_health()["last_error"]


class TestStreakVsTotal:
    """Consecutive is the signal; a long run may absorb isolated failures."""

    def test_success_clears_the_streak_but_not_the_total(self):
        c = _collector(max_failures=3)
        c._record_failure("generation", RuntimeError("boom"))
        c._record_failure("generation", RuntimeError("boom"))
        c._record_success()

        assert c._consecutive_failures == 0
        assert c._total_failures == 2
        assert c.is_healthy()

    def test_isolated_failures_never_trip_the_limit(self):
        c = _collector(max_failures=3)
        for _ in range(20):
            c._record_failure("generation", RuntimeError("transient"))
            c._record_success()
        assert c.is_healthy()
        assert c._total_failures == 20

    def test_an_unbroken_streak_trips_the_limit(self):
        c = _collector(max_failures=3)
        for _ in range(2):
            c._record_failure("generation", RuntimeError("wedged"))
        assert c.is_healthy()
        c._record_failure("generation", RuntimeError("wedged"))
        assert not c.is_healthy()

    def test_a_non_positive_limit_disables_the_check(self):
        """Opting out must be possible, and must be explicit."""
        c = _collector(max_failures=0)
        for _ in range(100):
            c._record_failure("generation", RuntimeError("boom"))
        assert c.is_healthy()


class TestTrainerAborts:
    """The trainer half: an unhealthy collector must end the run, not hang it."""

    class _Handle:
        def __init__(self, health: dict):
            self._health = health

        class _Remote:
            def __init__(self, health):
                self._health = health

            def remote(self):
                return self._health

        @property
        def check_health(self):
            return self._Remote(self._health)

    @pytest.fixture
    def no_ray(self, monkeypatch):
        import dockyard_rl.algorithms.grpo as grpo

        monkeypatch.setattr(grpo.ray, "get", lambda obj, **kw: obj)

    def test_a_healthy_collector_does_not_raise(self, no_ray):
        c = _collector()
        _raise_if_collector_unhealthy(self._Handle(c.check_health()))

    def test_an_unhealthy_collector_raises(self, no_ray):
        c = _collector(max_failures=2)
        c._record_failure("generation", RuntimeError("cuda oom"))
        c._record_failure("generation", RuntimeError("cuda oom"))

        with pytest.raises(RuntimeError, match="stopped producing trajectories"):
            _raise_if_collector_unhealthy(self._Handle(c.check_health()))

    def test_the_message_names_the_cause(self, no_ray):
        """A hang gave no diagnosis; the replacement must give a usable one."""
        c = _collector(max_failures=1)
        c._record_failure("teacher_scoring", ValueError("shape mismatch"))

        with pytest.raises(RuntimeError) as exc:
            _raise_if_collector_unhealthy(self._Handle(c.check_health()))

        message = str(exc.value)
        assert "teacher_scoring" in message
        assert "ValueError" in message
        assert "shape mismatch" in message


class TestHealthReportShape:
    def test_every_field_the_trainer_reads_is_present(self):
        c = _collector()
        health = c.check_health()
        for key in (
            "healthy", "consecutive_failures", "total_failures",
            "max_generation_failures", "failures_by_stage",
            "failures_by_type", "last_error",
        ):
            assert key in health, f"check_health() is missing {key!r}"

    def test_the_breakdowns_are_copies(self):
        """A Ray return crosses a process boundary; aliasing internal state is a trap."""
        c = _collector()
        c._record_failure("generation", RuntimeError("boom"))
        health = c.check_health()
        health["failures_by_stage"]["generation"] = 999
        assert c._failures_by_stage["generation"] == 1
