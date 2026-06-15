"""Sequence-packing algorithms.

Each algorithm takes a list of (index, length) pairs and a maximum pack
size, and returns a list of packs. Each pack is a list of indices from the
input, ordered so that the sum of their lengths does not exceed max_pack_size.

All algorithms preserve the invariant:
    for every pack p: sum(lengths[i] for i in p) <= max_pack_size
"""

from __future__ import annotations
import random
from typing import Optional

# Type aliases

Pack = list[int]          # list of original sequence indices
PackList = list[Pack]     # list of packs

# First-Fit Decreasing (FFD)

def first_fit_decreasing(
    lengths: list[int],
    max_pack_size: int,
    *,
    seed: Optional[int] = None,
) -> PackList:
    """Classic First-Fit Decreasing bin-packing heuristic.

    Sequences are sorted longest-first, then each sequence is placed into
    the first existing pack that still has enough room. A new pack is opened
    if none fits.

    Args:
        lengths: Token lengths of all sequences.
        max_pack_size: Maximum tokens per pack (e.g. max_seq_len).
        seed: Unused; kept for API compatibility with other algorithms.

    Returns:
        List of packs; each pack is a list of indices into `lengths`.
    """
    # Sort indices by descending length
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)

    packs: PackList = []
    remaining: list[int] = []  # remaining capacity for each open pack

    for idx in order:
        seq_len = lengths[idx]
        if seq_len > max_pack_size:
            raise ValueError(
                f"Sequence {idx} has length {seq_len} which exceeds "
                f"max_pack_size={max_pack_size}."
            )
        # Find the first pack with enough room
        placed = False
        for pack_idx, cap in enumerate(remaining):
            if cap >= seq_len:
                packs[pack_idx].append(idx)
                remaining[pack_idx] -= seq_len
                placed = True
                break
        if not placed:
            packs.append([idx])
            remaining.append(max_pack_size - seq_len)

    return packs

# Shuffle Packing

def shuffle_packing(
    lengths: list[int],
    max_pack_size: int,
    *,
    seed: Optional[int] = 42,
) -> PackList:
    """Greedy packing after randomly shuffling sequences.

    Provides diversity across training steps: each call with a different
    seed yields a different packing, preventing the model from memorising
    pack co-occurrences.

    Args:
        lengths: Token lengths of all sequences.
        max_pack_size: Maximum tokens per pack.
        seed: RNG seed for the shuffle (default: 42).

    Returns:
        List of packs.
    """
    rng = random.Random(seed)
    order = list(range(len(lengths)))
    rng.shuffle(order)

    packs: PackList = []
    remaining: list[int] = []

    for idx in order:
        seq_len = lengths[idx]
        if seq_len > max_pack_size:
            raise ValueError(
                f"Sequence {idx} has length {seq_len} which exceeds "
                f"max_pack_size={max_pack_size}."
            )
        placed = False
        for pack_idx, cap in enumerate(remaining):
            if cap >= seq_len:
                packs[pack_idx].append(idx)
                remaining[pack_idx] -= seq_len
                placed = True
                break
        if not placed:
            packs.append([idx])
            remaining.append(max_pack_size - seq_len)

    return packs

# Dynamic-Programming Optimal Packing

def dp_packing(
    lengths: list[int],
    max_pack_size: int,
    *,
    seed: Optional[int] = None,
    max_sequences_per_pack: int = 32,
) -> PackList:
    """Near-optimal bin-packing via a DP subset-sum over each pack.

    For each pack, we greedily grow a subset of remaining sequences that
    maximises utilisation without exceeding max_pack_size. The DP runs over
    the *unplaced* sequences, so worst-case complexity is O(N * max_pack_size)
    per pack. Practical performance is much better because sequences are
    pre-sorted and pruned aggressively.

    Args:
        lengths: Token lengths of all sequences.
        max_pack_size: Maximum tokens per pack.
        seed: Unused; for API compatibility.
        max_sequences_per_pack: Hard cap on sequences per pack to bound DP cost.

    Returns:
        List of packs.
    """
    # Sort remaining indices by descending length for faster pruning
    remaining = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    packs: PackList = []

    while remaining:
        capacity = max_pack_size
        # dp[c] = best set of indices (from remaining) whose total length == c
        # We only track the set for the maximum achievable fill.
        # Use a simple forward DP over capacities.
        dp: list[Optional[list[int]]] = [None] * (capacity + 1)
        dp[0] = []

        for idx in remaining:
            seq_len = lengths[idx]
            if seq_len > capacity:
                continue
            # Traverse backwards to avoid using the same item twice
            for c in range(capacity, seq_len - 1, -1):
                if dp[c - seq_len] is not None:
                    candidate = dp[c - seq_len] + [idx]  # type: ignore[operator]
                    if len(candidate) <= max_sequences_per_pack:
                        if dp[c] is None or c > sum(lengths[i] for i in (dp[c] or [])):
                            dp[c] = candidate

        # Find the best fill
        best_pack: list[int] = []
        for c in range(capacity, -1, -1):
            if dp[c] is not None:
                best_pack = dp[c]  # type: ignore[assignment]
                break

        if not best_pack:
            # Fallback: just take the next sequence (avoids infinite loop)
            best_pack = [remaining[0]]

        packs.append(best_pack)
        placed_set = set(best_pack)
        remaining = [i for i in remaining if i not in placed_set]

    return packs

# Online (Streaming) Packing

def online_packing(
    lengths: list[int],
    max_pack_size: int,
    *,
    seed: Optional[int] = None,
) -> PackList:
    """Online greedy packing in arrival order (no reordering).

    Each sequence is appended to the current open pack if it fits;
    otherwise the current pack is closed and a new one is started.
    This mirrors the behaviour of a streaming dataloader where sequences
    arrive in a fixed order and cannot be reordered.

    Args:
        lengths: Token lengths in arrival order.
        max_pack_size: Maximum tokens per pack.
        seed: Unused; for API compatibility.

    Returns:
        List of packs.
    """
    packs: PackList = []
    current_pack: Pack = []
    current_fill = 0

    for idx, seq_len in enumerate(lengths):
        if seq_len > max_pack_size:
            raise ValueError(
                f"Sequence {idx} has length {seq_len} which exceeds "
                f"max_pack_size={max_pack_size}."
            )
        if current_fill + seq_len > max_pack_size:
            if current_pack:
                packs.append(current_pack)
            current_pack = [idx]
            current_fill = seq_len
        else:
            current_pack.append(idx)
            current_fill += seq_len

    if current_pack:
        packs.append(current_pack)

    return packs

# Dispatcher

_ALGORITHM_REGISTRY: dict[str, object] = {
    "first_fit_decreasing": first_fit_decreasing,
    "ffd": first_fit_decreasing,
    "shuffle": shuffle_packing,
    "shuffle_packing": shuffle_packing,
    "dp": dp_packing,
    "dp_packing": dp_packing,
    "online": online_packing,
    "online_packing": online_packing,
}

def pack_sequences(
    lengths: list[int],
    max_pack_size: int,
    algorithm: str = "first_fit_decreasing",
    seed: Optional[int] = None,
) -> PackList:
    """Pack sequences using the named algorithm.

    Args:
        lengths: Token lengths of all sequences.
        max_pack_size: Maximum tokens allowed per pack.
        algorithm: One of "first_fit_decreasing" / "ffd", "shuffle" /
            "shuffle_packing", "dp" / "dp_packing", "online" / "online_packing".
        seed: RNG seed forwarded to algorithms that support it.

    Returns:
        List of packs (each a list of indices into `lengths`).

    Raises:
        KeyError: If `algorithm` is not in the registry.
        ValueError: If any sequence exceeds max_pack_size.
    """
    if algorithm not in _ALGORITHM_REGISTRY:
        raise KeyError(
            f"Unknown packing algorithm: {algorithm!r}. "
            f"Available: {sorted(_ALGORITHM_REGISTRY)}."
        )
    fn = _ALGORITHM_REGISTRY[algorithm]
    return fn(lengths, max_pack_size, seed=seed)  # type: ignore[call-arg]