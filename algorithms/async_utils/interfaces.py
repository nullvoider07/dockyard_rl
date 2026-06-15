"""Async RL interfaces."""

from typing import Any, Optional, Protocol

class ReplayBufferProtocol(Protocol):
    """Interface for the replay buffer used in async RL training."""

    def add(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
        target_weight_version: int,
    ) -> str:
        """Add a per-prompt trajectory group with metadata.

        Args:
            trajectory:            Data dict for the trajectory group.
            weight_version:        Version of model weights used for generation.
            target_weight_version: Version of model weights this trajectory is
                                   intended to train on.

        Returns:
            "success" if added, "full" if the buffer is at capacity.
        """
        ...

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample per-prompt trajectory groups for the current training step.

        Only returns trajectories with
        target_weight_version == current_weight_version.
        Returns None if insufficient trajectories are available, stalling
        training until the remaining trajectories are generated.  This
        ensures no trajectory loses its last chance to be used for its
        intended training step.

        Returns:
            Dict with 'trajectories' and 'avg_trajectory_age' keys,
            or None if insufficient data.
        """
        ...

    def evict(self) -> None:
        """Evict old trajectories."""
        ...

    def size(self) -> int:
        """Return current buffer size."""
        ...

    def clear(self) -> None:
        """Clear the buffer."""
        ...