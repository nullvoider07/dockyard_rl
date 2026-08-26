"""The linear-CE fusion entry point refuses what it cannot do (#2833 groundwork).

`from_parallel_hidden_states_to_logprobs` fuses the output projection into the
chunked log-softmax so the `[B, S, V]` logits are never materialized. It is
fully implemented, with its own chunked autograd Function and a hand-written
backward, and **nothing in the tree calls it**.

Wiring it is a port, not a connection: the implementation is sequence-first
`[S, B, H]` and sequence-parallel sharded (it chunks `shape[0]` and all-gathers
dim 0), while dockyard's DTensor workers hand HF models `[B, S]` and get
`[B, S, H]` back with the vocab sharded at the output projection. Until that
port happens, the entry point should say so rather than accept a wrong-shaped
tensor.

Two traps were in the signature itself: `chunk_size` defaulted to `None` while
the implementation divides by it, so the documented default raised TypeError
several frames down; and `runtime_gather_output` was accepted and read by
nothing.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from dockyard_rl.distributed import model_utils


def _hidden(seq=4, batch=2, hidden=8) -> torch.Tensor:
    return torch.randn(seq, batch, hidden)


def _target(batch=2, seq=4) -> torch.Tensor:
    return torch.randint(0, 8, (batch, seq))


class TestTheSignatureNoLongerLies:
    def test_chunk_size_is_required_and_positional(self):
        params = inspect.signature(
            model_utils.from_parallel_hidden_states_to_logprobs
        ).parameters

        assert params["chunk_size"].default is inspect.Parameter.empty
        assert params["chunk_size"].annotation is int

    def test_the_dropped_parameter_is_gone(self):
        params = inspect.signature(
            model_utils.from_parallel_hidden_states_to_logprobs
        ).parameters

        assert "runtime_gather_output" not in params

    @pytest.mark.parametrize("bad", [None, 0, -1])
    def test_a_non_positive_chunk_size_is_refused_by_name(self, bad):
        with pytest.raises(ValueError, match="chunk_size"):
            model_utils.from_parallel_hidden_states_to_logprobs(
                _hidden(),
                torch.randn(8, 8),
                _target(),
                0,
                8,
                None,  # type: ignore[arg-type]
                bad,  # type: ignore[arg-type]
            )


class TestTheLayoutContractIsEnforced:
    def test_a_batch_first_hidden_state_is_refused(self):
        """The likeliest mis-wiring: passing what the HF forward returns."""
        batch_first = torch.randn(2, 4, 8)

        with pytest.raises(ValueError, match=r"\[S, B, H\]"):
            model_utils.from_parallel_hidden_states_to_logprobs(
                batch_first.mean(dim=1),  # [2, 8] -- 2D, not [S, B, H]
                torch.randn(8, 8),
                _target(),
                0,
                8,
                None,  # type: ignore[arg-type]
                2,
            )

    def test_the_layout_is_documented_where_a_caller_will_read_it(self):
        doc = model_utils.from_parallel_hidden_states_to_logprobs.__doc__ or ""

        assert "[S, B, H]" in doc
        assert "no call site" in doc


class TestItIsStillUnwired:
    def test_nothing_calls_it_yet(self):
        """Pins the premise. When #2833 lands a caller, this test goes.

        It is here so the contract above is understood as groundwork rather
        than as coverage of a live path.
        """
        import pathlib

        root = pathlib.Path(model_utils.__file__).resolve().parents[1]
        callers = [
            path
            for path in root.rglob("*.py")
            if "from_parallel_hidden_states_to_logprobs(" in path.read_text()
            and path.name not in {"model_utils.py", "test_linear_ce_fusion_contract.py"}
        ]

        assert callers == [], f"it now has callers: {callers}"
