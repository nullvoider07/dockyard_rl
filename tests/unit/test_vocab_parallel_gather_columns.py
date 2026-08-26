"""Gathering the top-k student columns without the full-vocab log-softmax (#3314).

`_direct_topk_kl` renormalizes both sides inside the K-subset, and the
full-vocab partition function cancels in that renormalization:

    log_softmax(z)[cols] - logsumexp(log_softmax(z)[cols]) == log_softmax(z[cols])

so the student never needed a `[B, T, V]` log-softmax -- all-gathered across TP
-- to read `k` of its columns. These tests pin the identity itself, then the TP
gather that exploits it.

The TP path runs on a single-rank gloo group: the SUM all-reduce is a no-op
there, so the shard-masking arithmetic is exercised without a second process.
The multi-rank split is checked by simulating the shards directly.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from dockyard_rl.distributed.model_utils import (
    vocab_parallel_full_log_softmax,
    vocab_parallel_gather_columns,
)

_B, _T, _V = 2, 3, 16


@pytest.fixture(scope="module")
def group():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29593")
    created = not dist.is_initialized()
    if created:
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    yield dist.group.WORLD
    if created:
        dist.destroy_process_group()


def _logits(requires_grad: bool = False) -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randn(_B, _T, _V, dtype=torch.float32, requires_grad=requires_grad)


_COLS = torch.tensor([11, 2, 7, 15], dtype=torch.long)


class TestTheCancellationThisRestsOn:
    """If this identity fails, the optimization is not value-preserving."""

    @pytest.mark.parametrize("temperature", [1.0, 0.7, 2.5])
    def test_subset_renormalized_full_log_softmax_equals_subset_log_softmax(
        self, temperature
    ):
        logits = _logits()

        full = torch.log_softmax(logits / temperature, dim=-1)[..., _COLS]
        renormalized = full - torch.logsumexp(full, dim=-1, keepdim=True)
        direct = torch.log_softmax(logits[..., _COLS] / temperature, dim=-1)

        torch.testing.assert_close(renormalized, direct)


class TestSingleRankGather:
    def test_matches_plain_indexing_without_a_group(self):
        logits = _logits()

        got = vocab_parallel_gather_columns(logits, _COLS)

        torch.testing.assert_close(got, logits[..., _COLS])

    def test_matches_plain_indexing_on_a_single_rank_group(self, group):
        logits = _logits()

        got = vocab_parallel_gather_columns(logits, _COLS, tp_group=group)

        torch.testing.assert_close(got, logits[..., _COLS])

    def test_column_order_follows_the_request(self):
        logits = _logits()
        reversed_cols = _COLS.flip(0)

        got = vocab_parallel_gather_columns(logits, reversed_cols)

        torch.testing.assert_close(got, got.flip(-1).flip(-1))
        torch.testing.assert_close(got, logits[..., reversed_cols])

    def test_gradient_reaches_only_the_requested_columns(self):
        logits = _logits(True)

        vocab_parallel_gather_columns(logits, _COLS).sum().backward()

        assert logits.grad is not None
        touched = logits.grad.abs().sum(dim=(0, 1)) > 0
        assert torch.equal(touched.nonzero().flatten(), _COLS.sort().values)

    def test_a_two_dimensional_column_tensor_is_refused(self):
        with pytest.raises(ValueError, match="1-D"):
            vocab_parallel_gather_columns(_logits(), _COLS.view(2, 2))


class TestTensorParallelSplit:
    """The shard mask and SUM assembly, simulated across ranks in one process."""

    def _shard(self, logits: torch.Tensor, rank: int, world: int) -> torch.Tensor:
        width = _V // world
        return logits[..., rank * width : (rank + 1) * width]

    def _rank_contribution(
        self, shard: torch.Tensor, rank: int, world: int
    ) -> torch.Tensor:
        """What one rank contributes before the SUM all-reduce."""
        width = _V // world
        start = rank * width
        in_range = (_COLS >= start) & (_COLS < start + width)
        local = (_COLS - start).clamp(min=0, max=width - 1)
        return shard[..., local] * in_range.to(shard.dtype)

    @pytest.mark.parametrize("world", [2, 4])
    def test_the_rank_contributions_sum_to_the_full_gather(self, world):
        logits = _logits()

        assembled = sum(
            self._rank_contribution(self._shard(logits, r, world), r, world)
            for r in range(world)
        )

        torch.testing.assert_close(assembled, logits[..., _COLS])

    @pytest.mark.parametrize("world", [2, 4])
    def test_every_column_is_claimed_by_exactly_one_rank(self, world):
        """A column claimed twice would be double-counted by the SUM."""
        width = _V // world
        claims = torch.zeros(_COLS.numel(), dtype=torch.long)
        for rank in range(world):
            start = rank * width
            claims += ((_COLS >= start) & (_COLS < start + width)).long()

        assert torch.equal(claims, torch.ones_like(claims))


class TestTheLossUsesIt:
    def test_direct_topk_kl_gathers_columns_instead_of_the_full_vocab(self):
        """The wire. Reverting to the full log-softmax + slice fails here."""
        import inspect

        from dockyard_rl.algorithms.loss.loss_functions import (
            CrossTokenizerDistillationLossFn,
        )

        source = inspect.getsource(CrossTokenizerDistillationLossFn._direct_topk_kl)

        assert "vocab_parallel_gather_columns(" in source
        assert "vocab_parallel_full_log_softmax(" not in source

    def test_the_full_vocab_paths_still_use_the_full_log_softmax(self):
        """Two legitimate modes: averaged_logits and the gold path need all of it."""
        import inspect

        from dockyard_rl.algorithms.loss.loss_functions import (
            CrossTokenizerDistillationLossFn,
        )

        for method in ("_direct_full_vocab_kl", "_compute_gold"):
            source = inspect.getsource(
                getattr(CrossTokenizerDistillationLossFn, method)
            )
            assert "vocab_parallel_full_log_softmax(" in source, method
