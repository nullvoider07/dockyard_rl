from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PackingMetrics:
    """Accumulates and reports sequence-packing efficiency statistics.

    Attributes:
        total_sequences: Total number of original sequences considered.
        total_tokens: Total tokens across all sequences.
        total_padding: Total padding tokens added during packing.
        total_packs: Total number of packs produced.
        max_pack_size: Token capacity of each pack (i.e. max_seq_len).
        sequence_length_histogram: Optional dict mapping sequence-length bucket
            to count, for debugging skewed distributions.
    """

    total_sequences: int = 0
    total_tokens: int = 0
    total_padding: int = 0
    total_packs: int = 0
    max_pack_size: int = 0
    sequence_length_histogram: dict[int, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Derived properties
    # ------------------------------------------------------------------ #

    @property
    def packing_efficiency(self) -> float:
        """Fraction of pack capacity filled by real tokens (0–1)."""
        capacity = self.total_packs * self.max_pack_size
        if capacity == 0:
            return 0.0
        return self.total_tokens / capacity

    @property
    def padding_rate(self) -> float:
        """Fraction of pack capacity wasted on padding (0–1)."""
        capacity = self.total_packs * self.max_pack_size
        if capacity == 0:
            return 0.0
        return self.total_padding / capacity

    @property
    def avg_sequences_per_pack(self) -> float:
        if self.total_packs == 0:
            return 0.0
        return self.total_sequences / self.total_packs

    @property
    def avg_tokens_per_sequence(self) -> float:
        if self.total_sequences == 0:
            return 0.0
        return self.total_tokens / self.total_sequences

    # ------------------------------------------------------------------ #
    # Accumulation helpers
    # ------------------------------------------------------------------ #

    def update(
        self,
        sequences: list[int],
        packs: list[list[int]],
        max_pack_size: int,
    ) -> None:
        """Incorporate one batch of sequences and their resulting packs.

        Args:
            sequences: List of sequence lengths (tokens per sequence).
            packs: List of packs; each pack is a list of sequence lengths
                   whose sum ≤ max_pack_size.
            max_pack_size: Token capacity per pack.
        """
        self.total_sequences += len(sequences)
        self.total_tokens += sum(sequences)
        self.total_packs += len(packs)
        self.max_pack_size = max_pack_size

        for pack in packs:
            packed_tokens = sum(pack)
            self.total_padding += max_pack_size - packed_tokens

        # Update histogram with 128-token buckets
        bucket_size = 128
        for seq_len in sequences:
            bucket = (seq_len // bucket_size) * bucket_size
            self.sequence_length_histogram[bucket] = (
                self.sequence_length_histogram.get(bucket, 0) + 1
            )

    def to_log_dict(self, prefix: str = "packing/") -> dict[str, float]:
        """Return a flat dict suitable for logging."""
        return {
            f"{prefix}efficiency": self.packing_efficiency,
            f"{prefix}padding_rate": self.padding_rate,
            f"{prefix}avg_seqs_per_pack": self.avg_sequences_per_pack,
            f"{prefix}avg_tokens_per_seq": self.avg_tokens_per_sequence,
            f"{prefix}total_packs": float(self.total_packs),
            f"{prefix}total_sequences": float(self.total_sequences),
        }

    def reset(self) -> None:
        """Reset all accumulators."""
        self.total_sequences = 0
        self.total_tokens = 0
        self.total_padding = 0
        self.total_packs = 0
        self.sequence_length_histogram.clear()

def compute_packing_metrics(
    sequences: list[int],
    packs: list[list[int]],
    max_pack_size: int,
    prefix: str = "packing/",
) -> dict[str, float]:
    """One-shot helper that creates a PackingMetrics, updates it, and returns a log dict.

    Args:
        sequences: Original sequence lengths.
        packs: Resulting packs (each a list of sequence lengths).
        max_pack_size: Token capacity per pack.
        prefix: Metric-key prefix for the returned dict.

    Returns:
        Flat dict of packing statistics ready for logging.
    """
    m = PackingMetrics()
    m.update(sequences, packs, max_pack_size)
    return m.to_log_dict(prefix=prefix)