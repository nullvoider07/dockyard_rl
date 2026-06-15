"""BatchedDataDict: typed tensor/array container for distributed RL training.

BatchedDataDict is the canonical data transport object throughout the
Dockyard RL pipeline.  It carries named tensors or numpy arrays through:

  - Rollout collection (sandbox → replay buffer)
  - Policy forward passes (batch → logprobs, values)
  - GRPO advantage / reward computation
  - Data-parallel sharding (DP splitting across worker groups)

Design constraints
------------------
- Must be serialisable by Ray's object store (pickle-compatible).
- Must support zero-copy slicing where possible (numpy / torch views).
- Must tolerate heterogeneous value types: torch.Tensor, np.ndarray, list,
  scalar — with graceful fallback for non-tensor types.
- All batch operations must be deterministic across ranks.
- No GPU transfer happens inside this class; callers own device placement.
"""

from __future__ import annotations

import logging
from typing import Any, cast, Iterable, Iterator, Mapping, Optional, Sequence, TypedDict, TypeVar, Generic

import numpy as np

logger = logging.getLogger(__name__)

# Optional torch import for device utilities; not a hard dependency.  BatchedDataDict can be
# BatchedDataDict works without PyTorch for sandbox / CPU-only containers.
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# Optional PackedTensor (multimodal carrier). It is a deferred dependency only
# present when the VLM data path is exercised; on the text-only path the lookup
# fails and every helper below treats values as non-multimodal. Cached after the
# first lookup so the hot slicing/iteration paths pay the import cost once.
_PACKED_TENSOR_CLS: Any = None
_PACKED_TENSOR_LOOKED_UP = False

def _packed_tensor_cls() -> Any:
    global _PACKED_TENSOR_CLS, _PACKED_TENSOR_LOOKED_UP
    if not _PACKED_TENSOR_LOOKED_UP:
        _PACKED_TENSOR_LOOKED_UP = True
        try:
            from dockyard_rl.data.multimodal_utils import PackedTensor
            _PACKED_TENSOR_CLS = PackedTensor
        except ImportError:
            _PACKED_TENSOR_CLS = None
    return _PACKED_TENSOR_CLS

def _is_packed_tensor(v: Any) -> bool:
    cls = _packed_tensor_cls()
    return cls is not None and isinstance(v, cls)

def _is_tensor(v: Any) -> bool:
    return _TORCH_AVAILABLE and isinstance(v, torch.Tensor) # type: ignore

def _is_array(v: Any) -> bool:
    return isinstance(v, np.ndarray)

def _batch_dim(v: Any) -> int | None:
    """Return the batch (first) dimension size, or None for scalars."""
    if _is_tensor(v) or _is_array(v):
        return v.shape[0] if v.ndim > 0 else None
    if _is_packed_tensor(v):
        # After flattened_concat the PackedTensor holds one entry per sample, so
        # its length is the batch (first) dimension.
        return len(v)
    if isinstance(v, list):
        return len(v)
    return None

def _slice_value(v: Any, idx: slice | int | list[int]) -> Any:
    """Slice v along its first dimension."""
    if _is_tensor(v) or _is_array(v):
        return v[idx]
    if _is_packed_tensor(v):
        # PackedTensor is sample-indexed (one entry per sample); select by row
        # index. A bare int yields a length-1 PackedTensor rather than dropping
        # the batch dim, since a single sample's images cannot be unwrapped here.
        if isinstance(idx, slice):
            indices = list(range(len(v)))[idx]
        elif isinstance(idx, int):
            indices = [idx]
        else:
            indices = list(idx)
        return v.slice(indices)
    if isinstance(v, list):
        if isinstance(idx, slice):
            return v[idx]
        # Handle if an index array/list is passed to a standard Python list
        if isinstance(idx, list):
            return [v[i] for i in idx]
        return v[idx]
    return v

def _concat_values(vs: list[Any]) -> Any:
    """Concatenate a list of values along the first dimension."""
    if not vs:
        raise ValueError("Cannot concat empty list.")
    if _is_tensor(vs[0]):
        assert torch is not None
        return torch.cat(vs, dim=0)
    if _is_array(vs[0]):
        return np.concatenate(vs, axis=0)
    if _is_packed_tensor(vs[0]):
        # Each element is one batch's sample-indexed PackedTensor; concat extends
        # the underlying list so the result stays sample-indexed (len == total
        # samples) for downstream per-sample slicing.
        return _packed_tensor_cls().concat(cast("list[Any]", vs))
    if isinstance(vs[0], list):
        out = []
        for v in vs:
            out.extend(v)
        return out
    # Scalar or unknown: return list of values.
    return vs

# Recognised multimodal model-input keys (HF vision/audio processor outputs).
# Forwarded alongside input_ids for VLM batches; absent in text-only batches.
_MULTIMODAL_KEYS: tuple[str, ...] = (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "image_sizes",
    "image_embeds",
    "video_embeds",
    "second_per_grid_ts",
    "pixel_attention_mask",
    "input_features",
    "feature_attention_mask",
)

_T = TypeVar("_T")

class SequencePackingArgs(TypedDict):
    """Configuration settings for sequence packing.

    Pass this to ``shard_by_batch_size()`` to preprocess batches for sequence packing.
    """

    max_tokens_per_microbatch: int
    input_key: str
    input_lengths_key: str
    algorithm: str
    # pad each sequence to a multiple of this value (for CP/TP alignment)
    sequence_length_pad_multiple: int

class DynamicBatchingArgs(TypedDict):
    """Configuration settings for dynamic batching.

    Pass this to ``shard_by_batch_size()`` to preprocess batches for dynamic batching.
    """

    # Each microbatch contains at most this many tokens
    max_tokens_per_microbatch: int
    # Round each microbatch's sequence length to this multiple
    sequence_length_round: int
    # The key in the data dict that specifies the input ids
    input_key: str
    # The key in the data dict that specifies the sequence length per datum
    input_lengths_key: str

# BatchedDataDict
class BatchedDataDict(Generic[_T]):
    """A dict-like container of named tensors / arrays with a shared batch dim.

    All values must share the same first (batch) dimension size, or be
    scalars / non-batched metadata.  Batch-dimension size is inferred
    lazily from the first batched value found.

    Keys
    ----
    Standard RL field names used across the pipeline:

      "input_ids"          — Token ids for the prompt+response.
      "attention_mask"     — Padding mask.
      "labels"             — Target token ids (may equal input_ids shifted).
      "log_probs"          — Per-token log probabilities from the actor.
      "ref_log_probs"      — Log probs from the frozen reference policy.
      "values"             — Critic value estimates.
      "rewards"            — Scalar reward per episode.
      "advantages"         — GRPO / GAE advantages.
      "returns"            — Discounted returns.
      "response_lengths"   — Number of response tokens per sample.
      "prompt_lengths"     — Number of prompt tokens per sample.
      "episode_ids"        — Unique episode identifiers.
    """

    def __init__(
        self,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Initialise from an optional mapping.

        Args:
            data: Initial key → value mapping.  Copied shallowly; values
                  are not cloned.
        """
        self._data: dict[str, Any] = dict(data) if data else {}
        self._batch_size: int | None = None
        # Sharding metadata attached by shard_by_batch_size() in the
        # sequence-packing path; read by the packable-sequence microbatch
        # iterator. None for ordinary (un-packed) batches.
        self._packing_args: Optional[SequencePackingArgs] = None
        if self._data:
            self._batch_size = self._infer_batch_size()

    # Dict-like interface
    def __setitem__(self, key: str, value: Any) -> None:
        dim = _batch_dim(value)
        if dim is not None:
            if self._batch_size is None:
                self._batch_size = dim
            elif dim != self._batch_size:
                raise ValueError(
                    f"Cannot insert {key!r} with batch size {dim}; "
                    f"existing batch size is {self._batch_size}."
                )
        self._data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return self._batch_size if self._batch_size is not None else 0

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        field_shapes = {}
        for k, v in self._data.items():
            if _is_tensor(v) or _is_array(v):
                field_shapes[k] = tuple(v.shape)
            elif isinstance(v, list):
                field_shapes[k] = (len(v),)
            else:
                field_shapes[k] = type(v).__name__
        return (
            f"BatchedDataDict(batch_size={self._batch_size}, "
            f"fields={field_shapes})"
        )

    # Accessors
    @property
    def batch_size(self) -> int:
        if self._batch_size is None:
            raise RuntimeError(
                "batch_size is not set — BatchedDataDict is empty or contains "
                "only scalar values."
            )
        return self._batch_size

    @property
    def size(self) -> int:
        """Number of samples along the batch dimension (0 if unset).

        Unlike ``batch_size`` this never raises, so callers can guard on
        ``data.size > 0`` for possibly-empty batches.
        """
        return self._batch_size if self._batch_size is not None else 0

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def values(self) -> Iterable[Any]:
        return self._data.values()

    def items(self) -> Iterable[tuple[str, Any]]:
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def pop(self, key: str, *args: Any) -> Any:
        v = self._data.pop(key, *args)
        # Recalculate batch size in case the popped key was the only batched
        # value — rare but must be correct.
        self._batch_size = self._infer_batch_size()
        return v

    def update(self, other: "BatchedDataDict | dict[str, Any]") -> None:
        """Update this dict in-place from another BatchedDataDict or plain dict."""
        items = other.items() if isinstance(other, BatchedDataDict) else other.items()
        for k, v in items:
            self[k] = v

    def copy(self) -> "BatchedDataDict":
        """Shallow copy — values are not cloned."""
        return BatchedDataDict(dict(self._data))

    # Slicing and splitting
    def select(self, idx: slice | int | list[int]) -> "BatchedDataDict":
        """Return a new BatchedDataDict with values sliced along dim 0.

        Args:
            idx: A slice, integer index, or list of indices.

        Returns:
            New BatchedDataDict containing the selected samples.
        """
        return BatchedDataDict(
            {k: _slice_value(v, idx) for k, v in self._data.items()}
        )

    def select_indices(self, indices: Any) -> "BatchedDataDict":
        """Return a new BatchedDataDict containing only the given row indices.

        Args:
            indices: An iterable of integer row indices. Accepts a Python list,
                     numpy array, or torch.Tensor (normalised to a list).

        Returns:
            New BatchedDataDict with the selected samples, in the given order.
        """
        if _is_tensor(indices) or _is_array(indices):
            index_list = [int(i) for i in indices.tolist()]
        else:
            index_list = [int(i) for i in indices]
        return self.select(index_list)

    def get_batch(self, batch_idx: int, batch_size: int) -> "BatchedDataDict":
        """Return the ``batch_idx``-th contiguous chunk of ``batch_size`` rows.

        Args:
            batch_idx:  Zero-based chunk index.
            batch_size: Number of rows per chunk.

        Returns:
            New BatchedDataDict spanning rows
            ``[batch_idx * batch_size : (batch_idx + 1) * batch_size]``.
        """
        start = batch_idx * batch_size
        end = start + batch_size
        return self.select(slice(start, end))

    def make_microbatch_iterator(
        self, microbatch_size: int
    ) -> "Iterator[BatchedDataDict]":
        """Yield contiguous micro-batches of ``microbatch_size`` rows.

        The final micro-batch carries the remainder when the batch size is not
        an exact multiple of ``microbatch_size``.

        Args:
            microbatch_size: Number of rows per micro-batch.

        Yields:
            BatchedDataDict slices spanning the full batch in order.
        """
        if microbatch_size <= 0:
            raise ValueError(
                f"microbatch_size must be positive, got {microbatch_size}."
            )
        total = self.size
        for start in range(0, total, microbatch_size):
            yield self.select(slice(start, min(start + microbatch_size, total)))

    def repeat_interleave(self, num_repeats: int) -> "BatchedDataDict":
        """Repeat each sample ``num_repeats`` times, consecutively.

        Mirrors ``torch.repeat_interleave`` along the batch dimension: a batch
        ``[a, b]`` with ``num_repeats=2`` becomes ``[a, a, b, b]``. Non-batched
        (scalar / metadata) values are passed through unchanged.

        Args:
            num_repeats: Number of consecutive repeats per sample.

        Returns:
            New BatchedDataDict with ``batch_size * num_repeats`` samples.
        """
        if num_repeats <= 0:
            raise ValueError(f"num_repeats must be positive, got {num_repeats}.")
        new_data: dict[str, Any] = {}
        for k, v in self._data.items():
            if _batch_dim(v) is None:
                new_data[k] = v
            elif _is_tensor(v):
                assert torch is not None
                new_data[k] = v.repeat_interleave(num_repeats, dim=0)
            elif _is_array(v):
                new_data[k] = np.repeat(v, num_repeats, axis=0)
            elif _is_packed_tensor(v):
                # Sample-indexed PackedTensor: repeat each sample's entry in place
                # so the result stays aligned with the repeated batch dimension.
                cls = _packed_tensor_cls()
                new_data[k] = cls(
                    [t for t in v.tensors for _ in range(num_repeats)],
                    v.dim_to_pack,
                )
            elif isinstance(v, list):
                new_data[k] = [elem for elem in v for _ in range(num_repeats)]
            else:
                new_data[k] = v
        return BatchedDataDict(new_data)

    def get_multimodal_dict(
        self, as_tensors: bool = True, device: Any = None
    ) -> dict[str, Any]:
        """Return the multimodal model-input fields present in this batch.

        Multimodal inputs (image/video/audio processor outputs) are carried
        under recognised keys (as sample-indexed ``PackedTensor``s) and forwarded
        alongside ``input_ids`` to the model. Text-only batches carry none of
        these keys, so the result is empty.

        Args:
            as_tensors: If True (policy-worker forward), each PackedTensor is
                        materialised into the single concatenated tensor the model
                        expects, optionally moved to ``device``. If False (threading
                        into train / logprob data), the PackedTensor is returned
                        intact so subsequent per-sample sharding and microbatching
                        stay sample-aligned; the worker materialises each microbatch.
            device:     If given (and as_tensors is True), move materialised tensors
                        to this device — used when assembling model-forward kwargs.

        Returns:
            Dict of the recognised multimodal keys present in this batch.
        """
        out: dict[str, Any] = {}
        for k in _MULTIMODAL_KEYS:
            if k not in self._data:
                continue
            v = self._data[k]
            if _is_packed_tensor(v):
                if as_tensors:
                    # Worker forward: materialise the sample-indexed PackedTensor
                    # into the single concatenated tensor the model expects,
                    # optionally moved to `device`.
                    out[k] = (
                        v.as_tensor(device) if device is not None else v.as_tensor()
                    )
                else:
                    # Transport form (threading into train / logprob data): keep
                    # the PackedTensor intact so later per-sample sharding and
                    # microbatching stay sample-aligned; the worker re-materialises
                    # each microbatch via the as_tensors=True branch above.
                    out[k] = v
            elif not as_tensors and _is_tensor(v):
                out[k] = v.tolist()
            elif as_tensors and device is not None and _is_tensor(v):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out

    def split(self, n: int) -> list["BatchedDataDict"]:
        """Split evenly into n chunks along the batch dimension.

        If batch_size is not divisible by n, the last chunk receives the
        remainder.

        Args:
            n: Number of chunks.

        Returns:
            List of n BatchedDataDicts.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}.")
        bs   = self.batch_size
        size = bs // n
        chunks = []
        for i in range(n):
            start = i * size
            end   = (i + 1) * size if i < n - 1 else bs
            chunks.append(self.select(slice(start, end)))
        return chunks

    def split_for_dp(
        self,
        dp_size: int,
        drop_remainder: bool = True,
    ) -> list["BatchedDataDict"]:
        """Split the batch across data-parallel ranks.

        Each DP rank receives batch_size // dp_size samples.  If
        drop_remainder=True, any trailing samples that don't form a full
        per-rank batch are silently discarded.  This is the standard
        behaviour for RL training where we prefer uniform batch sizes.

        Args:
            dp_size:        Number of data-parallel ranks.
            drop_remainder: Drop samples that don't divide evenly.

        Returns:
            List of dp_size BatchedDataDicts, one per DP rank.

        Raises:
            ValueError: If batch_size < dp_size or drop_remainder=False
                        and batch_size % dp_size != 0.
        """
        bs = self.batch_size
        per_rank = bs // dp_size
        if per_rank == 0:
            raise ValueError(
                f"batch_size={bs} is smaller than dp_size={dp_size}. "
                "Not enough data to assign one sample per rank."
            )
        if not drop_remainder and bs % dp_size != 0:
            raise ValueError(
                f"batch_size={bs} is not divisible by dp_size={dp_size} "
                "and drop_remainder=False."
            )
        total = per_rank * dp_size
        trimmed = self.select(slice(0, total)) if total < bs else self
        return trimmed.split(dp_size)

    # Concatenation / stacking
    @staticmethod
    def concat(
        dicts: Sequence["BatchedDataDict"],
    ) -> "BatchedDataDict":
        """Concatenate multiple BatchedDataDicts along the batch dimension.

        All dicts must have identical key sets.

        Args:
            dicts: Sequence of BatchedDataDicts to concatenate.

        Returns:
            A new BatchedDataDict with all samples concatenated.
        """
        if not dicts:
            return BatchedDataDict()
        if len(dicts) == 1:
            return dicts[0].copy()

        all_keys = set(dicts[0].keys())
        for i, d in enumerate(dicts[1:], 1):
            if set(d.keys()) != all_keys:
                raise ValueError(
                    f"Key mismatch at index {i}: "
                    f"{set(d.keys())} != {all_keys}"
                )

        merged: dict[str, Any] = {}
        for key in all_keys:
            values = [d[key] for d in dicts]
            merged[key] = _concat_values(values)

        return BatchedDataDict(merged)

    @classmethod
    def from_batches(
        cls,
        batches: "Sequence[Mapping[str, Any] | BatchedDataDict[Any]]",
        pad_value_dict: Optional[Mapping[str, Any]] = None,
    ) -> "BatchedDataDict":
        """Stack a list of batches into a single BatchedDataDict.

        For each key, values are combined along the batch dimension:
          - 1-D tensors are concatenated.
          - 2-D tensors are right-padded to the longest sequence in the batch.
          - 3-D tensors ([B, S, K]) are padded along the sequence dim (1), keeping
            the feature dim, then concatenated on the batch dim.
          - numpy arrays are concatenated along axis 0.
          - list values are flattened (extended).
        Padding uses 0 by default, or a per-key value from ``pad_value_dict``.

        This mirrors the upstream sharding contract so the per-DP-shard outputs
        produced by generation workers can be recombined.

        Args:
            batches: A list of mappings (or BatchedDataDicts), each a batch of data.
            pad_value_dict: Optional mapping of key -> padding value (default 0).

        Returns:
            A new instance of the calling class containing the stacked data.
        """
        if not batches:
            return cls()
        pad_value_dict = pad_value_dict or {}
        stacked: dict[str, Any] = {}

        for k in sorted(batches[0]):
            values = [item[k] for item in batches]
            first = values[0]

            if isinstance(first, list):
                stacked[k] = [elem for sublist in values for elem in sublist]
            elif _is_tensor(first):
                assert torch is not None
                pad_value = pad_value_dict.get(k, 0)
                if all(t.ndim == 1 for t in values):
                    stacked[k] = torch.cat(values)
                elif first.ndim == 3:
                    # Pad along the sequence dim (1), keep the feature dim, concat on batch.
                    max_seq_len = max(t.shape[1] for t in values)
                    padded = [
                        torch.nn.functional.pad(
                            t,
                            (0, 0, 0, max_seq_len - t.shape[1]),
                            mode="constant",
                            value=pad_value,
                        )
                        for t in values
                    ]
                    stacked[k] = torch.cat(padded, dim=0)
                else:
                    rows = [row for t in values for row in t]
                    stacked[k] = torch.nn.utils.rnn.pad_sequence(
                        rows, batch_first=True, padding_value=pad_value
                    )
            elif _is_array(first):
                stacked[k] = np.concatenate(values, axis=0)
            elif _is_packed_tensor(first):
                # Sample-indexed PackedTensors: concat extends the underlying
                # per-sample list so the merged batch stays sample-aligned.
                stacked[k] = _packed_tensor_cls().concat(cast("list[Any]", values))
            else:
                raise NotImplementedError(
                    f"from_batches cannot stack values of type {type(first)!r} "
                    f"for key {k!r}; provide a tensor, numpy array, or list."
                )

        return cls(stacked)

    def shard_by_batch_size(
        self,
        shards: int,
        batch_size: Optional[int] = None,
        allow_uneven_shards: bool = False,
        dynamic_batching_args: Optional[DynamicBatchingArgs] = None,
        sequence_packing_args: Optional[SequencePackingArgs] = None,
    ) -> "list[SlicedDataDict] | tuple[list[SlicedDataDict], list[int]]":
        """Shard a batch into ``shards`` parts, striped across batch_size chunks.

        The batch is first divided into chunks of ``batch_size`` (the whole batch if
        None), each chunk is split into ``shards`` equal parts, and the sub-shards are
        aggregated by position. For example, data [A A B B C C D D] with batch_size=2,
        shards=2 yields [[A B C D], [A B C D]]. With ``allow_uneven_shards`` the last
        shard may be smaller (and ``batch_size`` must be None).

        Args:
            shards: Number of shards to divide each chunk into.
            batch_size: Size of each initial chunk (defaults to the whole batch).
            allow_uneven_shards: Allow the last shard to be smaller than the others.
            dynamic_batching_args / sequence_packing_args: Not supported in this build
                (see NotImplementedError below).

        Returns:
            A list of ``shards`` SlicedDataDicts.
        """
        if dynamic_batching_args is not None:
            raise NotImplementedError(
                "shard_by_batch_size() does not support dynamic_batching_args in "
                "this build. Dynamic microbatching is unported; set "
                "policy.dynamic_batching.enabled=false (the shipped default)."
            )
        if sequence_packing_args is not None:
            # Sequence-packing path: sort by length, shard into contiguous blocks,
            # and return the inverse permutation so the gathered (sorted) results
            # can be restored to the original order via reorder_data().
            return self._shard_by_batch_size_packed(shards, sequence_packing_args)

        if allow_uneven_shards:
            assert batch_size is None, (
                "batch_size must be None if allow_uneven_shards is True"
            )

        # Determine the (single, consistent) batch size across all values.
        batch_sizes = set()
        for val in self._data.values():
            if _is_tensor(val):
                batch_sizes.add(val.size(0))
            elif _is_array(val):
                batch_sizes.add(val.shape[0])
            else:
                batch_sizes.add(len(val))
        assert len(batch_sizes) == 1, (
            "Batch sizes are not the same across the batch, found sizes: "
            + f"[{','.join(str(s) for s in batch_sizes)}]"
        )
        total_batch_size = int(batch_sizes.pop())
        # Resolve the (non-Optional) chunk size used for all sharding math.
        chunk_size: int = total_batch_size if batch_size is None else batch_size

        assert total_batch_size % chunk_size == 0, (
            f"Total batch size ({total_batch_size}) is not a multiple of "
            f"batch_size ({chunk_size})"
        )
        if not allow_uneven_shards:
            assert chunk_size % shards == 0, (
                f"Batch size ({chunk_size}) is not a multiple of shards ({shards})"
            )

        num_chunks = total_batch_size // chunk_size
        shard_size = (
            (chunk_size + shards - 1) // shards
            if allow_uneven_shards
            else chunk_size // shards
        )

        aggregated_shards: list[SlicedDataDict] = [
            SlicedDataDict() for _ in range(shards)
        ]
        for shard_idx in range(shards):
            shard_ranges: list[tuple[int, int]] = []
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                shard_start = chunk_start + shard_idx * shard_size
                shard_end = chunk_start + (shard_idx + 1) * shard_size
                if allow_uneven_shards:
                    # Cap at the total for the trailing shard.
                    shard_start = min(shard_start, total_batch_size)
                    shard_end = min(shard_end, total_batch_size)
                if shard_start < shard_end:
                    shard_ranges.append((shard_start, shard_end))

            for k, v in self._data.items():
                if _is_tensor(v):
                    assert torch is not None
                    slices = [v[start:end] for start, end in shard_ranges]
                    aggregated_shards[shard_idx][k] = (
                        torch.cat(slices, dim=0) if slices else v[:0]
                    )
                elif _is_array(v):
                    slices = [v[start:end] for start, end in shard_ranges]
                    aggregated_shards[shard_idx][k] = (
                        np.concatenate(slices, axis=0) if slices else v[:0]
                    )
                elif _is_packed_tensor(v):
                    # Sample-indexed PackedTensor: select each shard range by row
                    # index and concat the parts back into one PackedTensor (an
                    # empty shard has no samples, so an empty list is correct).
                    parts = [
                        v.slice(list(range(start, end)))
                        for start, end in shard_ranges
                    ]
                    aggregated_shards[shard_idx][k] = (
                        _packed_tensor_cls().concat(parts) if parts else []
                    )
                else:
                    shard_values: list[Any] = []
                    for start, end in shard_ranges:
                        shard_values.extend(v[i] for i in range(start, end))
                    aggregated_shards[shard_idx][k] = shard_values

        return aggregated_shards

    def _packable_lengths(self, lengths_key: str) -> list[int]:
        """Return per-sample sequence lengths as a list of ints."""
        v = self._data[lengths_key]
        if _is_tensor(v) or _is_array(v):
            return [int(x) for x in v.tolist()]
        return [int(x) for x in v]

    def _shard_by_batch_size_packed(
        self,
        shards: int,
        sequence_packing_args: SequencePackingArgs,
    ) -> tuple[list["SlicedDataDict"], list[int]]:
        """Sequence-packing shard: sort by length, shard, return inverse perm.

        The batch is stably sorted by sequence length (ascending) for cross-shard
        load balancing, sharded into contiguous blocks via the ordinary un-packed
        path, and tagged with the packing args so each shard's
        ``make_microbatch_iterator_for_packable_sequences`` can bin its sequences.
        The returned ``unsorted_data_indices`` is the inverse permutation: after
        the per-shard results are gathered (which reconstructs the sorted order),
        ``reorder_data(unsorted_data_indices)`` restores the original order.
        """
        lengths = self._packable_lengths(sequence_packing_args["input_lengths_key"])
        n = len(lengths)
        sort_order = sorted(range(n), key=lambda i: lengths[i])
        inverse = [0] * n
        for sorted_pos, original_idx in enumerate(sort_order):
            inverse[original_idx] = sorted_pos

        reordered = self.select(sort_order)
        # Reuse the un-packed contiguous-block sharding on the sorted data
        # (no packing args passed, so this returns the plain list branch).
        shard_list = cast(
            "list[SlicedDataDict]",
            reordered.shard_by_batch_size(shards, batch_size=None),
        )
        for shard in shard_list:
            shard._packing_args = sequence_packing_args
        return shard_list, inverse

    def reorder_data(self, reorder_indices: Sequence[int]) -> None:
        """Reindex the batch rows in place by ``reorder_indices``.

        After ``new[i] = old[reorder_indices[i]]`` for every batched value. Used
        to undo the length-sort applied by the sequence-packing / dynamic shard
        path once per-shard results have been gathered. Non-batched (scalar /
        metadata) values are left unchanged.

        Args:
            reorder_indices: Permutation mapping new positions to old indices.
        """
        idx = [int(i) for i in reorder_indices]
        for k, v in self._data.items():
            if _batch_dim(v) is None:
                continue
            self._data[k] = _slice_value(v, idx)
        self._batch_size = self._infer_batch_size()

    def _compute_packs(self) -> list[list[int]]:
        """Bin this (packing-tagged) batch's sequences into token-budget packs."""
        args = getattr(self, "_packing_args", None)
        if args is None:
            raise RuntimeError(
                "Packable-sequence iteration requires a batch produced by "
                "shard_by_batch_size(sequence_packing_args=...)."
            )
        # Deferred import: keeps the data-container free of a data/ dependency
        # at module load time.
        from dockyard_rl.data.packing.algorithms import pack_sequences

        lengths = self._packable_lengths(args["input_lengths_key"])
        # Fixed seed: the iterator and its _len companion are called separately
        # and must produce identical packs.
        return pack_sequences(
            lengths,
            args["max_tokens_per_microbatch"],
            args.get("algorithm", "first_fit_decreasing"),
            seed=0,
        )

    def make_microbatch_iterator_for_packable_sequences(
        self,
    ) -> "Iterator[BatchedDataDict]":
        """Yield micro-batches whose sequences fit a token budget per pack.

        Each yielded micro-batch is a row-subset of this shard (padded
        ``input_ids`` + ``input_lengths``); the worker concatenates the pack into
        a single packed sequence before the forward pass.
        """
        for pack in self._compute_packs():
            yield self.select_indices(pack)

    def get_microbatch_iterator_for_packable_sequences_len(self) -> tuple[int, int]:
        """Return ``(num_micro_batches, max_packed_tokens)`` for the packs.

        ``max_packed_tokens`` is the largest total token count across packs.
        Computed with the same deterministic packing as the iterator.
        """
        args = getattr(self, "_packing_args", None)
        if args is None:
            raise RuntimeError(
                "Packable-sequence iteration requires a batch produced by "
                "shard_by_batch_size(sequence_packing_args=...)."
            )
        lengths = self._packable_lengths(args["input_lengths_key"])
        packs = self._compute_packs()
        max_packed = max(
            (sum(lengths[i] for i in pack) for pack in packs), default=0
        )
        return len(packs), max_packed

    def make_microbatch_iterator_with_dynamic_shapes(
        self,
    ) -> "Iterator[BatchedDataDict]":
        """Dynamic-microbatch iteration — unported in this build.

        Dynamic batching is disabled in all shipped configs
        (``policy.dynamic_batching.enabled=false``); sequence packing is the
        supported token-budget path.
        """
        raise NotImplementedError(
            "make_microbatch_iterator_with_dynamic_shapes() is unported; set "
            "policy.dynamic_batching.enabled=false and use sequence_packing."
        )

    def get_microbatch_iterator_dynamic_shapes_len(self) -> int:
        """Companion length for the dynamic-shapes iterator — unported."""
        raise NotImplementedError(
            "get_microbatch_iterator_dynamic_shapes_len() is unported; set "
            "policy.dynamic_batching.enabled=false and use sequence_packing."
        )

    # Device / dtype utilities
    def to(
        self,
        device: Any,
        non_blocking: bool = True,
    ) -> "BatchedDataDict":
        """Move all torch.Tensor values to a device.

        Non-tensor values are left unchanged.  numpy arrays are not
        converted — caller should convert explicitly if needed.

        Args:
            device:       torch device string or torch.device.
            non_blocking: Passed to torch.Tensor.to().

        Returns:
            A new BatchedDataDict with tensors on the target device.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not available; cannot call BatchedDataDict.to()."
            )
        new_data: dict[str, Any] = {}
        for k, v in self._data.items():
            if _is_tensor(v):
                new_data[k] = v.to(device, non_blocking=non_blocking)
            elif _is_packed_tensor(v):
                # PackedTensor wraps a list of tensors; move them too so the
                # multimodal values track the rest of the batch's device.
                new_data[k] = v.to(device)
            else:
                new_data[k] = v
        return BatchedDataDict(new_data)

    def pin_memory(self) -> "BatchedDataDict":
        """Pin all CPU torch.Tensor values in memory for fast H2D transfer."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not available; cannot call BatchedDataDict.pin_memory()."
            )
        new_data: dict[str, Any] = {}
        for k, v in self._data.items():
            if _is_tensor(v) and not v.is_cuda:
                try:
                    new_data[k] = v.pin_memory()
                except RuntimeError:
                    # Pinning fails for some tensor types (e.g. bool).
                    new_data[k] = v
            else:
                new_data[k] = v
        return BatchedDataDict(new_data)

    def to_numpy(self) -> "BatchedDataDict":
        """Convert all torch.Tensor values to numpy arrays (CPU, detached).

        Returns:
            A new BatchedDataDict with numpy arrays in place of tensors.
        """
        if not _TORCH_AVAILABLE:
            return self.copy()
        new_data: dict[str, Any] = {}
        for k, v in self._data.items():
            if _is_tensor(v):
                new_data[k] = v.detach().cpu().numpy()
            else:
                new_data[k] = v
        return BatchedDataDict(new_data)

    def to_torch(
        self,
        dtype: Any = None,
    ) -> "BatchedDataDict":
        """Convert all numpy array values to torch.Tensors.

        Args:
            dtype: Optional torch dtype for all converted tensors.

        Returns:
            A new BatchedDataDict with torch.Tensors in place of arrays.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not available; cannot call BatchedDataDict.to_torch()."
            )
        new_data: dict[str, Any] = {}
        for k, v in self._data.items():
            if _is_array(v):
                assert torch is not None
                t = torch.from_numpy(v)
                if dtype is not None:
                    t = t.to(dtype)
                new_data[k] = t
            else:
                new_data[k] = v
        return BatchedDataDict(new_data)

    # Serialisation helpers
    def __getstate__(self) -> dict:
        return {
            "_data": self._data,
            "_batch_size": self._batch_size,
            "_packing_args": getattr(self, "_packing_args", None),
        }

    def __setstate__(self, state: dict) -> None:
        self._data       = state["_data"]
        self._batch_size = state["_batch_size"]
        self._packing_args = state.get("_packing_args")

    # Internal helpers
    def _infer_batch_size(self) -> int | None:
        for v in self._data.values():
            dim = _batch_dim(v)
            if dim is not None:
                return dim
        return None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BatchedDataDict":
        """Construct from a plain dict (alias for the constructor)."""
        return cls(d)

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy as a plain dict."""
        return dict(self._data)

class SlicedDataDict(BatchedDataDict):
    """A BatchedDataDict that represents a slice / shard of a larger batch.

    A distinct type so call sites can differentiate full batches from
    sliced/sharded batches for type checking; carries no extra behaviour.
    """

    pass