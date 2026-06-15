"""N-dimensional rank layout with named axes for sharding, replication, and
data routing across TP / PP / DP / EP parallelism dimensions.

Example
-------
    layout = [[[0, 1, 2, 3], [4, 5, 6, 7]]]
    names  = ["dp", "pp", "tp"]
    # DP=1, PP=2, TP=4
    sharding = NamedSharding(layout, names)
    sharding.shape                      # {"dp": 1, "pp": 2, "tp": 4}
    sharding.get_ranks(dp=0, pp=1)      # NamedSharding over ranks [4,5,6,7]
    sharding.get_ranks_by_coord(pp=1)   # [4, 5, 6, 7]
    sharding.get_worker_coords(5)       # {"dp": 0, "pp": 1, "tp": 1}
"""

import numpy as np
from typing import Any, Sequence, Union

# Axis names that are replicated: every rank along these axes holds the same
# data. Used as the default ``axes`` for :meth:`NamedSharding.is_axis_zero`
# leader-rank gating — the replica leader is the worker at coord 0 on every
# replicated axis. Single source of truth; a typo in a caller's inline list
# would silently route around the leader gate.
REPLICATED_AXES: tuple[str, ...] = (
    "tensor_parallel",
    "context_parallel",
    "pipeline_parallel",
)


class NamedSharding:
    """N-dimensional arrangement of ranks with named axes.

    Facilitates data sharding, replication, and collection based on named
    parallelism axes (e.g. "dp", "pp", "tp", "ep").
    """

    def __init__(
        self,
        layout: Sequence[Any] | np.ndarray,
        names: list[str],
    ) -> None:
        """Initialise NamedSharding.

        Args:
            layout: Nested sequence (list-of-lists) or ndarray representing
                    the N-D rank layout.  All leaf values must be integer rank
                    IDs and must be unique across the layout.
            names:  Axis names ordered from outermost to innermost dimension.

        Raises:
            ValueError: Non-integer values, duplicate ranks, or shape/name
                        length mismatch.
        """
        try:
            initial = np.array(layout)
        except ValueError as exc:
            raise ValueError(
                f"Could not create NumPy array from layout: {exc}"
            ) from exc

        if not np.issubdtype(initial.dtype, np.integer):
            if not np.equal(np.mod(initial, 1), 0).all():
                raise ValueError("Layout must contain only integer rank IDs.")
            initial = initial.astype(np.int32)

        self._layout: np.ndarray = initial
        self._names: list[str] = list(names)

        if self._layout.ndim != len(self._names):
            raise ValueError(
                f"Number of layout dimensions ({self._layout.ndim}) must "
                f"match the number of names ({len(self._names)})."
            )

        unique, counts = np.unique(self._layout, return_counts=True)
        duplicates = unique[counts > 1]
        if duplicates.size > 0:
            raise ValueError(
                f"Duplicate ranks found in layout: {duplicates.tolist()}"
            )

        self._name_to_axis: dict[str, int] = {
            name: i for i, name in enumerate(self._names)
        }

    # Properties and accessors
    @property
    def shape(self) -> dict[str, int]:
        """Shape of the rank layout as a name → size mapping."""
        return {
            name: size
            for name, size in zip(self._names, self._layout.shape)
        }

    @property
    def names(self) -> list[str]:
        """Axis names (copy)."""
        return list(self._names)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._layout.ndim

    @property
    def size(self) -> int:
        """Total number of ranks."""
        return int(self._layout.size)

    @property
    def layout(self) -> np.ndarray:
        """Underlying NumPy rank array (copy)."""
        return self._layout.copy()

    # Lookup helpers
    def get_worker_coords(self, worker_id: int) -> dict[str, int]:
        """Return the axis coordinates of a given rank ID.

        Args:
            worker_id: Integer rank ID to look up.

        Returns:
            Dict mapping each axis name to its coordinate index for this rank.

        Raises:
            ValueError: If worker_id is not present in the layout.
        """
        indices = np.where(self._layout == worker_id)
        if not indices[0].size:
            raise ValueError(
                f"Worker ID {worker_id} not found in sharding layout."
            )
        return {
            axis_name: indices[i].item()
            for i, axis_name in enumerate(self._names)
        }

    @staticmethod
    def is_axis_zero(coords: dict[str, int], axes: Sequence[str]) -> bool:
        """Return True when ``coords`` is 0 on every axis in ``axes``.

        Replica-leader check shared with ``TQWorkerMixin._local_coords`` on
        the worker side; driver-side callers can pair it with
        ``get_worker_coords``. Axes absent from ``coords`` are treated as 0.
        """
        return all(coords.get(ax, 0) == 0 for ax in axes)

    def get_ranks_by_coord(self, **coords: int) -> list[int]:
        """Return all ranks matching the specified axis coordinates.

        Unspecified axes match all coordinates along that axis.

        Args:
            **coords: Axis-name → coordinate pairs to filter by.

        Returns:
            Sorted list of matching rank IDs (empty list if none match).

        Raises:
            ValueError: If an unknown axis name is supplied.
        """
        slicing: list[Any] = [slice(None)] * self.ndim

        for name, index in coords.items():
            if name not in self._name_to_axis:
                raise ValueError(
                    f"Invalid axis name {name!r}. "
                    f"Valid names: {self.names}"
                )
            axis_idx = self._name_to_axis[name]
            if not (0 <= index < self.shape[name]):
                return []
            slicing[axis_idx] = index

        matching = self._layout[tuple(slicing)]
        return sorted(np.unique(matching.flatten()).tolist())

    def get_ranks(self, **kwargs: int) -> Union["NamedSharding", int]:
        """Slice the layout along named axes, returning a sub-sharding.

        If all axes are specified, returns the single integer rank ID.

        Args:
            **kwargs: Axis-name → index pairs for the axes to slice.

        Returns:
            A new NamedSharding over the remaining axes, or an int when all
            axes are fully specified.

        Raises:
            ValueError / IndexError: Unknown axis name or out-of-bounds index.
        """
        indices: list[Any] = [slice(None)] * self.ndim
        specified: set[int] = set()

        for name, index in kwargs.items():
            if name not in self._name_to_axis:
                raise ValueError(
                    f"Invalid axis name {name!r}. "
                    f"Valid names: {self.names}"
                )
            if not (0 <= index < self.shape[name]):
                raise IndexError(
                    f"Index {index} out of bounds for axis {name!r} "
                    f"(size {self.shape[name]})."
                )
            axis_idx = self._name_to_axis[name]
            indices[axis_idx] = index
            specified.add(axis_idx)

        subset = self._layout[tuple(indices)]
        remaining_names = [
            n for i, n in enumerate(self._names) if i not in specified
        ]

        if not remaining_names:
            return int(subset.item())

        return NamedSharding(subset, remaining_names)

    def get_axis_index(self, name: str) -> int:
        """Return the numerical index of a named axis."""
        if name not in self._name_to_axis:
            raise ValueError(
                f"Invalid axis name {name!r}. Valid names: {self.names}"
            )
        return self._name_to_axis[name]

    def get_axis_size(self, name: str) -> int:
        """Return the size of a named axis."""
        return self.shape[name]

    # Dunder helpers
    def __repr__(self) -> str:
        shape_str = ", ".join(
            f"{name}={self.shape[name]}" for name in self.names
        )
        return (
            f"NamedSharding(shape=({shape_str}), "
            f"names={self.names}, layout={self._layout})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NamedSharding):
            return NotImplemented
        return (
            np.array_equal(self._layout, other._layout)
            and self._names == other._names
        )