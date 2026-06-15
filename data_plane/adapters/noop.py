"""In-memory ``DataPlaneClient`` adapter.

Behaves like a real adapter end-to-end (put → get → clear, consumption
counters, field-presence as the stage-done signal) but stores everything
in process memory. The ABC contract tests run against this implementation
so they don't require TransferQueue installed, and it doubles as a
selectable backend (``impl="noop"``) for single-process / CI / local
dry-runs of the data-plane trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from tensordict import TensorDict

from dockyard_rl.data_plane.codec import stack_or_nest as _stack_or_nest
from dockyard_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta


def _reject_non_tensor_leaves(td: TensorDict) -> None:
    """No pickle on the bus. Mirror of the TQ adapter check.

    Walk the leaves via ``keys()`` + indexed lookup rather than
    ``items()``, because some tensordict versions skip ``NonTensorData``
    entries from ``items(leaves_only=True)`` — they're "leaves" by
    structure but not tensor-typed, so they'd silently slip past a
    naive items() iteration.
    """
    bad = []
    for k in td.keys(include_nested=True, leaves_only=True):
        v = td.get(k)
        if not isinstance(v, torch.Tensor):
            bad.append(k)
    if bad:
        raise TypeError(
            f"put_samples received non-tensor leaves: {bad}. "
            "Tensorize via codec helpers, use `tags=` for primitives, "
            "or use the Ray object store for arbitrary Python objects."
        )


@dataclass
class _Partition:
    fields: list[str]
    num_samples: int
    consumer_tasks: list[str]
    grpo_group_size: int | None
    enums: dict[str, list[str]]
    rows: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    tags: dict[str, dict[str, Any]] = field(default_factory=dict)
    # per-task set of keys already returned by claim_meta (TQ ``mode='fetch'``)
    consumed: dict[str, set[str]] = field(default_factory=dict)


class NoOpDataPlaneClient(DataPlaneClient):
    """Reference in-memory implementation."""

    def __init__(self) -> None:
        self._partitions: dict[str, _Partition] = {}
        self._closed = False

    def register_partition(
        self,
        partition_id: str,
        fields: list[str],
        num_samples: int,
        consumer_tasks: list[str],
        grpo_group_size: int | None = None,
        enums: dict[str, list[str]] | None = None,
    ) -> None:
        self._partitions[partition_id] = _Partition(
            fields=list(fields),
            num_samples=int(num_samples),
            consumer_tasks=list(consumer_tasks),
            grpo_group_size=grpo_group_size,
            enums=dict(enums) if enums else {},
            consumed={t: set() for t in consumer_tasks},
        )

    def claim_meta(
        self,
        partition_id: str,
        task_name: str,
        required_fields: list[str],
        batch_size: int,
        dp_rank: int | None = None,
        blocking: bool = True,
        timeout_s: float = 60.0,
    ) -> KVBatchMeta:
        del blocking, timeout_s, dp_rank  # NoOp is single-process
        rec = self._partitions[partition_id]
        if task_name not in rec.consumed:
            raise KeyError(
                f"task {task_name!r} not registered as a consumer of "
                f"partition {partition_id!r}"
            )

        ready: list[str] = []
        seqs: list[int] = []
        for key, row in rec.rows.items():
            if key in rec.consumed[task_name]:
                continue
            if not all(f in row for f in required_fields):
                continue
            ready.append(key)
            tag = rec.tags.get(key, {})
            seqs.append(int(tag.get("input_lengths", 0)))
            if len(ready) >= batch_size:
                break

        rec.consumed[task_name].update(ready)
        return KVBatchMeta(
            partition_id=partition_id,
            task_name=task_name,
            sample_ids=ready,
            fields=list(required_fields),
            sequence_lengths=seqs if any(seqs) else None,
        )

    def get_data(
        self,
        meta: KVBatchMeta,
        select_fields: list[str] | None = None,
    ) -> TensorDict:
        fields = select_fields if select_fields is not None else meta.fields
        if fields is None:
            raise ValueError(
                "get_data requires either select_fields or meta.fields; "
                "fetching all fields silently is forbidden."
            )
        return self.get_samples(meta.sample_ids, meta.partition_id, list(fields))

    def check_consumption_status(
        self, partition_id: str, task_names: list[str]
    ) -> bool:
        rec = self._partitions[partition_id]
        for t in task_names:
            if t not in rec.consumed:
                return False
            if len(rec.consumed[t]) < len(rec.rows):
                return False
        return True

    def put_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> KVBatchMeta:
        rec = self._partitions[partition_id]
        if fields is not None:
            _reject_non_tensor_leaves(fields)
            for i, sid in enumerate(sample_ids):
                row = rec.rows.setdefault(sid, {})
                for fname in fields.keys():
                    val = fields[cast(str, fname)][i]
                    # Defense in depth — _reject_non_tensor_leaves can
                    # miss NonTensorData entries depending on the
                    # tensordict version's iteration semantics.
                    if not isinstance(val, torch.Tensor):
                        raise TypeError(
                            f"put_samples received non-tensor leaf "
                            f"{fname!r}: {type(val).__name__}. "
                            "Tensorize via codec helpers, use `tags=` "
                            "for primitives, or use the Ray object store "
                            "for arbitrary Python objects."
                        )
                    row[cast(str, fname)] = val.detach().clone()
        if tags is not None:
            for sid, tag in zip(sample_ids, tags):
                rec.tags.setdefault(sid, {}).update(tag)
        return KVBatchMeta(
            partition_id=partition_id,
            task_name=None,
            sample_ids=list(sample_ids),
            fields=cast(list[str], list(fields.keys())) if fields is not None else None,
            tags=[dict(t) for t in tags] if tags is not None else None,
        )

    def get_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str],
    ) -> TensorDict:
        rec = self._partitions[partition_id]
        if not sample_ids:
            return TensorDict({}, batch_size=(0,))

        out: dict[str, list[torch.Tensor]] = {f: [] for f in select_fields}
        for sid in sample_ids:
            row = rec.rows[sid]
            for f in select_fields:
                if f not in row:
                    raise KeyError(
                        f"field {f!r} not yet produced for sample_id {sid!r} "
                        f"in partition {partition_id!r}"
                    )
                out[f].append(row[f])

        stacked = {f: _stack_or_nest(out[f]) for f in select_fields}
        return TensorDict(stacked, batch_size=(len(sample_ids),))

    def clear_samples(self, sample_ids: list[str] | None, partition_id: str) -> None:
        rec = self._partitions.get(partition_id)
        if rec is None:
            return
        if sample_ids is None:
            rec.rows.clear()
            rec.tags.clear()
            for s in rec.consumed.values():
                s.clear()
            self._partitions.pop(partition_id, None)
            return
        for sid in sample_ids:
            rec.rows.pop(sid, None)
            rec.tags.pop(sid, None)
            for s in rec.consumed.values():
                s.discard(sid)

    def close(self) -> None:
        if self._closed:
            return
        self._partitions.clear()
        self._closed = True
