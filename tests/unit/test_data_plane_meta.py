"""KVBatchMeta pure-metadata transforms (Phase 5, slice 5.1) — CPU coverage.

The data-plane boundary (``interfaces``) defers the ``tensordict`` import under
TYPE_CHECKING, so ``KVBatchMeta`` and its pure transforms import and test
WITHOUT tensordict / transfer_queue installed (the dependency-light upgrade).
The tensor-moving adapters are covered separately and are device/dep-bound.
"""

from __future__ import annotations

import pytest

from dockyard_rl.data_plane.interfaces import KVBatchMeta


def _meta(n: int, *, with_lens: bool = True, with_tags: bool = False) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="train",
        task_name="logprob",
        sample_ids=[f"s{i}" for i in range(n)],
        fields=["input_ids"],
        sequence_lengths=list(range(1, n + 1)) if with_lens else None,
        tags=[{"std": float(i)} for i in range(n)] if with_tags else None,
    )


class TestConstruction:
    def test_size(self):
        assert _meta(3).size == 3

    def test_tags_must_align(self):
        with pytest.raises(ValueError, match="align 1:1"):
            KVBatchMeta(
                partition_id="p",
                task_name=None,
                sample_ids=["a", "b"],
                tags=[{"x": 1}],  # length 1 != 2
            )

    def test_tags_none_ok(self):
        m = KVBatchMeta(partition_id="p", task_name=None, sample_ids=["a"])
        assert m.tags is None and m.size == 1


class TestStampTags:
    def test_stamp_initializes_and_writes(self):
        m = _meta(3)
        assert m.tags is None
        m.stamp_tags({"std": [0.1, 0.2, 0.3], "wv": [7, 7, 7]})
        assert m.tags == [
            {"std": 0.1, "wv": 7},
            {"std": 0.2, "wv": 7},
            {"std": 0.3, "wv": 7},
        ]

    def test_stamp_wrong_length_raises(self):
        m = _meta(3)
        with pytest.raises(ValueError, match="expected 3"):
            m.stamp_tags({"std": [0.1, 0.2]})


class TestSubset:
    def test_reorders_and_filters(self):
        m = _meta(4, with_tags=True)
        sub = m.subset([3, 0])
        assert sub.sample_ids == ["s3", "s0"]
        assert sub.sequence_lengths == [4, 1]
        assert sub.tags == [{"std": 3.0}, {"std": 0.0}]
        # metadata preserved, original untouched
        assert sub.partition_id == "train" and sub.task_name == "logprob"
        assert m.sample_ids == ["s0", "s1", "s2", "s3"]

    def test_subset_without_lens_or_tags(self):
        m = _meta(3, with_lens=False)
        sub = m.subset([2, 1])
        assert sub.sample_ids == ["s2", "s1"]
        assert sub.sequence_lengths is None and sub.tags is None


class TestSlice:
    def test_contiguous_range(self):
        m = _meta(5, with_tags=True)
        sl = m.slice(1, 3)
        assert sl.sample_ids == ["s1", "s2"]
        assert sl.sequence_lengths == [2, 3]
        assert sl.tags == [{"std": 1.0}, {"std": 2.0}]


class TestConcat:
    def test_concat_appends(self):
        a = _meta(2, with_tags=True)
        b = a.subset([0])  # 1 row, same partition
        out = a.concat(b)
        assert out.sample_ids == ["s0", "s1", "s0"]
        assert out.sequence_lengths == [1, 2, 1]
        assert out.tags == [{"std": 0.0}, {"std": 1.0}, {"std": 0.0}]

    def test_concat_mismatched_partition_raises(self):
        a = _meta(1)
        b = KVBatchMeta(partition_id="other", task_name=None, sample_ids=["x"])
        with pytest.raises(ValueError, match="partition_ids must match"):
            a.concat(b)

    def test_concat_drops_lens_if_any_missing(self):
        a = _meta(2)  # has lens
        b = KVBatchMeta(partition_id="train", task_name=None, sample_ids=["z"])  # no lens
        out = a.concat(b)
        assert out.sample_ids == ["s0", "s1", "z"]
        assert out.sequence_lengths is None  # not all had lens
