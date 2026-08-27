"""The teacher contract: score a batch of trajectories, return per-token logprobs.

OPD's teachers are the one class of model dockyard never refits — loaded frozen
at startup and only ever asked for logprobs. That makes them the one fleet whose
implementation is genuinely interchangeable: a local DTensor worker group today,
a remote inference endpoint or an external API tomorrow. Nothing about the
consuming code needs to know which.

It did not read that way before this protocol existed. The OPD path reached into
the teacher object for four things that only a locally sharded worker group has:
its data-parallel axis size (to pad a sub-batch up to a multiple of it and trim
the pad rows back off), its packing mode, its sequence pad multiple, and a lock
serializing collectives across its workers. A remote teacher has none of them, so
the padding and the serialization move behind this boundary: callers hand over
rows and get one logprob row back per row in, and how those rows are laid out on
whatever hardware answers is the implementation's business.

What deliberately stays outside: the caller still pre-pads the *sequence* axis
when a scorer asks it to, because that padding is shared by every teacher scoring
the same flattened batch and cannot be decided per teacher.

The hard limit this contract makes explicit — an API-served teacher can supply
per-token logprobs but not full-vocabulary logit tensors, so a remote teacher can
drive the token-level OPD advantage and cannot drive a full-vocab KL. Any
full-vocab mode is a separate protocol, not a wider version of this one.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class TeacherScorer(Protocol):
    """A frozen scorer over student trajectories."""

    alias: str

    @property
    def requires_prepadded_input(self) -> bool:
        """Whether the caller must pre-pad the sequence axis for this scorer.

        ``False`` when the scorer re-pads its own forward (sequence packing, or a
        remote endpoint that tokenizes and batches on its own side).
        """
        ...

    @property
    def sequence_pad_multiple(self) -> int:
        """Divisor the caller pre-pads ``[B, S]`` inputs to when required."""
        ...

    def score(self, data: Any) -> torch.Tensor:
        """Per-token logprobs ``[B, S]`` for ``data``'s rows.

        One output row per input row, in input order. Any batch-axis padding the
        implementation needs to satisfy its own sharding is added and removed
        inside this call, so it is never observable to the caller -- a pad row
        that leaked out would shift every subsequent row's logprobs onto the
        wrong sample.
        """
        ...

    def shutdown(self) -> bool:
        """Release the scorer's resources. Idempotent."""
        ...


def select_rows(key: str, value: Any, row_indices: list[int]) -> Any:
    """Row-select one batch column onto a subset (or repetition) of its rows.

    Used on both sides of the teacher boundary: the caller slices the flattened
    batch down to the rows one teacher is routed, and the scorer repeats rows to
    reach a batch size its own sharding accepts. Both are row operations on
    sample-indexed columns, so both go through here.

    ``get_multimodal_dict(as_tensors=False)`` yields a ``PackedTensor`` (one
    entry per sample) or, for a plain tensor column, the list form -- both
    sample-indexed. Anything else is refused rather than forwarded whole: a
    column that cannot be row-selected would reach the teacher misaligned with
    the rows it is scoring.

    ``row_indices`` may repeat an index; that is how a pad row is produced.
    """
    slicer = getattr(value, "slice", None)
    if callable(slicer):
        return slicer(row_indices)
    if isinstance(value, torch.Tensor):
        return value[row_indices]
    if isinstance(value, (list, tuple)):
        return [value[i] for i in row_indices]
    raise TypeError(
        f"batch column {key!r} is a {type(value).__name__}, which cannot be "
        "row-selected for a teacher sub-batch"
    )
