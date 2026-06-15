"""Replay buffer for async GRPO training.

A single entry corresponds to 1 prompt repeated by
grpo.num_generations_per_prompt (required to compute per-prompt advantages).

Version semantics:
  weight_version        — version of model weights used to generate the trajectory
  target_weight_version — version of model weights this trajectory will train on

Example: weight_version=1, target_weight_version=4 means weights at step 1
were used for generation, and the trajectory will be consumed when the trainer
reaches step 4.
"""

import threading as _threading
from collections import Counter
from typing import Any, Optional
import ray
from dockyard_rl.algorithms.async_utils.interfaces import ReplayBufferProtocol

class ReplayBufferImpl(ReplayBufferProtocol):
    """Thread-safe replay buffer storing per-prompt trajectory groups."""

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")

        self.max_size = max_size
        self.trajectories:             list[dict[str, Any]] = []
        self.trajectory_versions:      list[int]            = []
        self.target_weight_versions:   list[int]            = []

        # Tracks the highest target_weight_version that has at least one
        # trajectory already generated (used by the collector to avoid
        # re-generating targets that are already covered).
        self.last_target_weight_already_generated: int = -1
        self._lock = _threading.Lock()

    # Write path
    def add(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
        target_weight_version: int,
    ) -> str:
        with self._lock:
            if len(self.trajectories) >= self.max_size:
                return "full"

            print("📌 ReplayBuffer.add: Adding trajectory")
            self.trajectories.append(trajectory)
            self.trajectory_versions.append(weight_version)
            self.target_weight_versions.append(target_weight_version)
            self.last_target_weight_already_generated = max(
                self.last_target_weight_already_generated,
                target_weight_version,
            )
            print(
                f"ReplayBuffer state: {len(self.trajectories)} groups, "
                f"versions={self.trajectory_versions}, "
                f"targets={self.target_weight_versions}, "
                f"last_target_weight_already_generated="
                f"{self.last_target_weight_already_generated}"
            )
            return "success"

    # Read path
    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            if not self.trajectories:
                return None

            total_trajectories = len(self.trajectories)
            print("📌 ReplayBuffer sampling debug:")
            print(f"   {current_weight_version=}, {max_age_steps=}")
            print(f"   {self.trajectory_versions=}")

            # Minimum valid trajectory version based on the age window.
            # max_age_steps=1 means trajectories from the last 1 step are valid.
            min_valid_version = max(0, current_weight_version - max_age_steps)
            print(f"   {min_valid_version=}")

            # Catch unexpected stale trajectories.
            old_trajectories = [
                v for v in self.trajectory_versions
                if v < min_valid_version
            ]
            if old_trajectories:
                raise ValueError(
                    f"Found {len(old_trajectories)} trajectories older than "
                    f"min_valid_version {min_valid_version}."
                )

            # All trajectories within the age window.
            valid_indices = [
                i
                for i, v in enumerate(self.trajectory_versions)
                if min_valid_version <= v <= current_weight_version
            ]
            print(
                f"   valid_indices: {len(valid_indices)}/{total_trajectories} "
                f"within age window"
            )
            if not valid_indices:
                print("No trajectories available for sampling.")
                return None

            if len(valid_indices) < num_prompt_groups:
                print(
                    f"Insufficient valid groups: have {len(valid_indices)}, "
                    f"need {num_prompt_groups}. Waiting for buffer to fill."
                )
                return None

            # Only select trajectories intended for the current training step;
            # ensures no trajectory loses its "last chance" for its target step.
            intended_indices = [
                i for i in valid_indices
                if self.target_weight_versions[i] == current_weight_version
            ]
            print(
                f"   🎯 Found {len(intended_indices)} trajectories intended "
                f"for current step {current_weight_version}"
            )

            if len(intended_indices) < num_prompt_groups:
                print(
                    f"   ⏸️ STALLING: Need {num_prompt_groups} trajectories "
                    f"for step {current_weight_version}, but only "
                    f"{len(intended_indices)} are ready"
                )
                return None

            # FIFO within same target version.
            selected: list[int] = intended_indices[:num_prompt_groups]
            print(
                f"   ✅ Selected {len(selected)} trajectories all intended "
                f"for step {current_weight_version}"
            )

            sampled_weights = [self.trajectory_versions[i] for i in selected]
            avg_trajectory_age = (
                current_weight_version
                - sum(sampled_weights) / len(sampled_weights)
            )
            print(
                f"✅ Selected counts by generation weight-version: "
                f"{Counter(sampled_weights)}"
            )
            print(f"📊 Average trajectory age: {avg_trajectory_age:.2f} steps")

            sampled_items = [self.trajectories[i] for i in selected]

            # Remove in reverse order to preserve correct indices.
            for idx in sorted(selected, reverse=True):
                self.trajectory_versions.pop(idx)
                self.target_weight_versions.pop(idx)
                self.trajectories.pop(idx)

            print(
                f"🗑️ Consumed and removed {len(selected)} groups from buffer, "
                f"old size: {total_trajectories}, "
                f"new size: {len(self.trajectories)}, "
                f"new target weight versions {self.target_weight_versions}"
            )

            return {
                "trajectories":        sampled_items,
                "avg_trajectory_age":  avg_trajectory_age,
            }

    # Utilities
    def evict(self) -> None:
        """Evict old trajectories (no-op — retained for interface compatibility)."""
        pass

    def size(self) -> int:
        with self._lock:
            return len(self.trajectories)

    def clear(self) -> None:
        with self._lock:
            self.trajectories.clear()
            self.trajectory_versions.clear()
            self.target_weight_versions.clear()

    def get_debug_info(self) -> dict:
        return {
            "total_trajectories":   len(self.trajectories),
            "trajectory_versions":  self.trajectory_versions,
            "target_weight_versions": self.target_weight_versions,
            "max_size":             self.max_size,
        }

    def get_last_target_weight_already_generated(self) -> int:
        with self._lock:
            return self.last_target_weight_already_generated

    def get_existing_target_weights(self) -> set[int]:
        with self._lock:
            return set(self.target_weight_versions)

@ray.remote  # pragma: no cover
class ReplayBuffer(ReplayBufferImpl):
    pass