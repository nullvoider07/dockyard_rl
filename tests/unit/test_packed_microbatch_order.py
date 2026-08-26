"""Packed micro-batches: which rows come back, in what order, and from where.

Sequence packing is `enabled: true` in the shipped configs, and three things
about it did not hold up:

  * every shipped config named `modified_first_fit_decreasing`, which is not in
    the packing registry, so `pack_sequences` raised `KeyError` at the first
    packed micro-batch -- after cluster spin-up, model load and a full rollout;
  * `make_microbatch_iterator_for_packable_sequences` yields bin-packer row
    subsets, so concatenating per-micro-batch outputs is in pack order, not
    shard-row order. The driver's `reorder_data(unsorted_data_indices)` only
    undoes the length sort applied before sharding, so it assumes the gathered
    rows are already in shard order. Row counts match either way and nothing
    errors, so a mismatch is silent;
  * the packs were yielded in packer order rather than heaviest first, which
    lets one DP rank meet its heavy pack at a different step index from another
    when every rank synchronises at each gradient all-reduce.

The first two are correctness and the third is scheduling, but they are one
batch because fixing the config alone would unmask the ordering bug.
"""

from __future__ import annotations

import pytest
import torch

from dockyard_rl.data.packing.algorithms import (
    available_algorithms,
    pack_sequences,
    validate_packing_algorithm,
)
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict


def _packed(lengths: list[int], max_tokens: int) -> BatchedDataDict:
    """A shard as `_shard_by_batch_size_packed` leaves it: length-sorted, tagged."""
    ordered = sorted(lengths)
    batch: BatchedDataDict = BatchedDataDict(
        {
            "input_ids": torch.arange(len(ordered) * 4).view(len(ordered), 4),
            "input_lengths": torch.tensor(ordered, dtype=torch.long),
        }
    )
    batch._packing_args = {  # type: ignore[attr-defined]
        "max_tokens_per_microbatch": max_tokens,
        "input_key": "input_ids",
        "input_lengths_key": "input_lengths",
        "algorithm": "first_fit_decreasing",
        "sequence_length_pad_multiple": 1,
    }
    return batch


class TestTheShippedAlgorithmNameResolves:
    def test_the_registry_is_reachable_and_non_empty(self):
        assert "first_fit_decreasing" in available_algorithms()

    def test_the_name_the_configs_used_is_not_an_algorithm(self):
        """Pins why the configs changed. If this is ever implemented, revisit."""
        assert "modified_first_fit_decreasing" not in available_algorithms()

    @pytest.mark.parametrize(
        "config",
        [
            "grpo_swe.yaml",
            "distillation.yaml",
            "sft.yaml",
            "xtoken_distillation.yaml",
        ],
    )
    def test_shipped_configs_name_a_registered_algorithm(self, config):
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[2]
        text = (root / "examples" / "configs" / config).read_text()
        names = re.findall(r'^\s*algorithm:\s*"([^"]+)"', text, flags=re.MULTILINE)

        assert names, f"{config} declares no packing algorithm"
        for name in names:
            assert name in available_algorithms(), (
                f"{config} names {name!r}, which pack_sequences would refuse"
            )


class TestTheAlgorithmIsValidatedAtConfigTime:
    def test_a_registered_name_passes_through(self):
        assert validate_packing_algorithm("ffd") == "ffd"

    def test_an_unregistered_name_is_refused_by_name(self):
        with pytest.raises(ValueError) as excinfo:
            validate_packing_algorithm("modified_first_fit_decreasing")

        message = str(excinfo.value)
        assert "policy.sequence_packing.algorithm" in message
        assert "first_fit_decreasing" in message  # the alternatives are listed

    def test_the_policy_validates_the_config_value(self):
        """The wire: Policy.__init__ must route the config through the check."""
        import inspect

        from dockyard_rl.models.policy import lm_policy

        source = inspect.getsource(lm_policy.Policy.__init__)
        assert "validate_packing_algorithm(" in source
        assert 'config["sequence_packing"]["algorithm"]' in source


class TestTheWorkerScattersBackToRowOrder:
    """`_restore_packed_row_order` is the consumer of the order accessor."""

    def _worker(self, enabled: bool):
        from dockyard_rl.models.policy.workers.dtensor_policy_worker import (
            DTensorPolicyWorkerImpl,
        )

        worker = DTensorPolicyWorkerImpl.__new__(DTensorPolicyWorkerImpl)
        worker.enable_seq_packing = enabled
        return worker

    def _round_trip(self, batch: BatchedDataDict, rows: torch.Tensor):
        """Outputs as the worker sees them: one row per packed index, in order."""
        return rows[batch.get_packable_sequence_order()].clone()

    def test_outputs_land_on_the_rows_they_were_computed_from(self):
        batch = _packed([10, 20, 30, 40, 55, 15, 25], 100)
        rows = torch.arange(7, dtype=torch.float32).unsqueeze(-1)
        as_computed = self._round_trip(batch, rows)

        restored = self._worker(True)._restore_packed_row_order(as_computed, batch)

        torch.testing.assert_close(restored, rows)

    def test_without_the_scatter_the_rows_are_misattributed(self):
        """Pins that the scatter is load-bearing, not defensive."""
        batch = _packed([10, 20, 30, 40, 55, 15, 25], 100)
        rows = torch.arange(7, dtype=torch.float32).unsqueeze(-1)

        as_computed = self._round_trip(batch, rows)

        assert not torch.equal(as_computed, rows)

    def test_it_is_a_no_op_when_packing_is_off(self):
        batch = _packed([10, 20, 30], 100)
        rows = torch.arange(3, dtype=torch.float32).unsqueeze(-1)

        out = self._worker(False)._restore_packed_row_order(rows, batch)

        assert out is rows

    def test_a_row_count_mismatch_is_refused(self):
        """Dummy micro-batches leaking into the output would land here."""
        batch = _packed([10, 20, 30], 100)
        too_many = torch.zeros(5, 1)

        with pytest.raises(RuntimeError, match="cover the shard exactly"):
            self._worker(True)._restore_packed_row_order(too_many, batch)

    @pytest.mark.parametrize(
        "method,next_method",
        [("get_logprobs", "score"), ("score", "get_topk_logits")],
    )
    def test_both_output_paths_scatter(self, method, next_method):
        """get_logprobs and score both return one row per input row.

        Read from the file rather than via inspect.getsource: the nvtx
        decorator is not functools.wraps'd, so getsource returns the wrapper.
        """
        import pathlib

        from dockyard_rl.models.policy.workers import dtensor_policy_worker

        text = pathlib.Path(dtensor_policy_worker.__file__).read_text()
        start = text.index(f"    def {method}(")
        end = text.index(f"    def {next_method}(", start)

        assert "_restore_packed_row_order(" in text[start:end], method


class TestPackOrderIsNotRowOrder:
    def test_the_packer_does_not_return_rows_in_index_order(self):
        """The premise. FFD walks longest-first, so packs interleave indices."""
        packs = pack_sequences([10, 20, 30, 40, 55], 100, "first_fit_decreasing", seed=0)
        flat = [i for pack in packs for i in pack]

        assert flat != sorted(flat), (
            "the packer happened to return index order; this test's premise, "
            "and the scatter it justifies, need rechecking"
        )

    def test_the_order_accessor_covers_every_row_exactly_once(self):
        batch = _packed([10, 20, 30, 40, 55], 100)

        order = batch.get_packable_sequence_order()

        assert sorted(order) == list(range(5))

    def test_the_order_accessor_matches_what_the_iterator_yields(self):
        """The accessor is only correct if it reproduces the iterator exactly."""
        batch = _packed([10, 20, 30, 40, 55, 15, 25], 100)

        from_iterator = [
            int(row)
            for micro in batch.make_microbatch_iterator_for_packable_sequences()
            for row in micro["input_ids"][:, 0] // 4
        ]

        assert batch.get_packable_sequence_order() == from_iterator

    def test_scattering_through_the_order_restores_row_alignment(self):
        """What the worker does, on a stand-in for its per-row outputs."""
        batch = _packed([10, 20, 30, 40, 55], 100)
        marker = torch.arange(5, dtype=torch.float32)

        concatenated = torch.cat(
            [
                marker[batch.get_packable_sequence_order()][i : i + 1]
                for i in range(5)
            ]
        )
        order = torch.as_tensor(batch.get_packable_sequence_order())
        restored = torch.empty_like(concatenated)
        restored[order] = concatenated

        torch.testing.assert_close(restored, marker)


class TestHeaviestPackFirst:
    def test_packs_are_yielded_heaviest_first(self):
        batch = _packed([10, 20, 30, 40, 55, 15, 25], 100)

        weights = [
            int(micro["input_lengths"].sum())
            for micro in batch.make_microbatch_iterator_for_packable_sequences()
        ]

        assert weights == sorted(weights, reverse=True), weights

    def test_ordering_does_not_change_the_partition(self):
        """Reordering must move packs, never their contents."""
        lengths = [10, 20, 30, 40, 55, 15, 25]
        batch = _packed(lengths, 100)
        ordered = sorted(lengths)

        packs = {
            tuple(sorted(pack))
            for pack in pack_sequences(ordered, 100, "first_fit_decreasing", seed=0)
        }
        yielded = {
            tuple(sorted(int(r) for r in micro["input_ids"][:, 0] // 4))
            for micro in batch.make_microbatch_iterator_for_packable_sequences()
        }

        assert yielded == packs

    def test_the_order_is_stable_across_calls(self):
        """The iterator, its _len companion and the order accessor must agree."""
        batch = _packed([10, 20, 30, 40, 55, 15, 25], 100)

        first = batch.get_packable_sequence_order()
        count, _max_tokens = batch.get_microbatch_iterator_for_packable_sequences_len()
        second = batch.get_packable_sequence_order()

        assert first == second
        assert count == len(
            list(batch.make_microbatch_iterator_for_packable_sequences())
        )
