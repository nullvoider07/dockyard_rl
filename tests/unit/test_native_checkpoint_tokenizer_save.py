"""#3504: tokenizer.save_pretrained ran on every rank against one path.

save_pretrained is not collective. It writes rank-independent filenames, so
every rank wrote the same files to the same directory, and it is non-atomic --
open(O_TRUNC), write, read back -- so N-1 redundant writers race the same
inodes. On a hard-mounted NFS share that presents as a hang at checkpoint time.

The discrimination that matters is the second class below: the dcp.save calls
in the same function are collective and rank-disjoint, so applying the same
rank-0 guard there would drop every non-zero rank's shard and deadlock the
collective. A fix that guarded "the saves" rather than "the non-collective
save" would be worse than the defect. These tests pin both halves.

torch.distributed is faked; no process group is created. The claim under test
is which ranks reach which call, which is exactly what a fake resolves.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from dockyard_rl.utils import native_checkpoint


class _FakeTokenizer:
    def __init__(self) -> None:
        self.saved_to: list[str] = []

    def save_pretrained(self, path: str) -> None:
        self.saved_to.append(path)


@pytest.fixture
def harness(monkeypatch):
    """Fake dcp.save and torch.distributed; return a recorder and a runner."""
    rec: dict[str, Any] = {"dcp_saves": [], "barriers": 0, "rank": 0, "initialized": True}

    monkeypatch.setattr(
        native_checkpoint.dcp,
        "save",
        lambda state, checkpoint_id=None: rec["dcp_saves"].append(checkpoint_id),
    )
    monkeypatch.setattr(native_checkpoint, "ModelState", lambda model: object())
    monkeypatch.setattr(
        native_checkpoint, "OptimizerState", lambda model, opt, sched: object()
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: rec["initialized"])
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rec["rank"])

    def _barrier(*args, **kwargs):
        rec["barriers"] += 1

    monkeypatch.setattr(torch.distributed, "barrier", _barrier)

    def run(rank: int, *, initialized: bool = True, tokenizer=None, tokenizer_path="/ck/tok"):
        rec["rank"] = rank
        rec["initialized"] = initialized
        native_checkpoint.save_checkpoint(
            model=torch.nn.Linear(2, 2),
            weights_path="/ck/weights",
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path if tokenizer is not None else None,
        )

    return rec, run


class TestOnlyRankZeroWritesTheTokenizer:
    def test_rank_zero_writes(self, harness):
        rec, run = harness
        tok = _FakeTokenizer()
        run(0, tokenizer=tok)
        assert tok.saved_to == ["/ck/tok"]

    @pytest.mark.parametrize("rank", [1, 2, 7])
    def test_other_ranks_do_not_write(self, harness, rank):
        rec, run = harness
        tok = _FakeTokenizer()
        run(rank, tokenizer=tok)
        assert tok.saved_to == []

    def test_single_process_still_writes(self, harness):
        """Not every run is distributed; an unguarded single process must save."""
        rec, run = harness
        tok = _FakeTokenizer()
        run(0, initialized=False, tokenizer=tok)
        assert tok.saved_to == ["/ck/tok"]

    def test_no_barrier_when_not_distributed(self, harness):
        rec, run = harness
        run(0, initialized=False, tokenizer=_FakeTokenizer())
        assert rec["barriers"] == 0


class TestCollectiveSavesAreNotGuarded:
    """Guarding these would deadlock and drop shards -- the opposite mistake."""

    @pytest.mark.parametrize("rank", [0, 1, 5])
    def test_every_rank_calls_dcp_save_for_weights(self, harness, rank):
        rec, run = harness
        run(rank, tokenizer=_FakeTokenizer())
        assert rec["dcp_saves"] == ["/ck/weights"]

    @pytest.mark.parametrize("rank", [0, 3])
    def test_every_rank_calls_dcp_save_for_the_optimizer(self, harness, monkeypatch, rank):
        rec, _ = harness
        rec["rank"] = rank
        model = torch.nn.Linear(2, 2)
        native_checkpoint.save_checkpoint(
            model=model,
            weights_path="/ck/weights",
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            optimizer_path="/ck/optim",
        )
        assert rec["dcp_saves"] == ["/ck/weights", "/ck/optim"]


class TestBarrierSynchronisesEveryRank:
    @pytest.mark.parametrize("rank", [0, 1, 4])
    def test_all_ranks_reach_the_barrier(self, harness, rank):
        """Asymmetric participation would hang the job, not just skew it."""
        rec, run = harness
        run(rank, tokenizer=_FakeTokenizer())
        assert rec["barriers"] == 1

    def test_no_barrier_when_no_tokenizer_is_saved(self, harness):
        """The barrier lives in the tokenizer branch; without it there is none."""
        rec, run = harness
        run(0, tokenizer=None)
        assert rec["barriers"] == 0
        assert rec["dcp_saves"] == ["/ck/weights"]


class TestArgumentValidationUnchanged:
    def test_tokenizer_without_path_still_raises(self, harness):
        rec, _ = harness
        with pytest.raises(ValueError, match="tokenizer_path must be provided"):
            native_checkpoint.save_checkpoint(
                model=torch.nn.Linear(2, 2),
                weights_path="/ck/weights",
                tokenizer=_FakeTokenizer(),
                tokenizer_path=None,
            )

    def test_optimizer_without_path_still_raises(self, harness):
        rec, _ = harness
        model = torch.nn.Linear(2, 2)
        with pytest.raises(ValueError, match="optimizer_path must be provided"):
            native_checkpoint.save_checkpoint(
                model=model,
                weights_path="/ck/weights",
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                optimizer_path=None,
            )
