# OSWorld actuation/observation backend interface.
#
# OSWorld's DesktopEnv bundles provisioning, environment setup, observation,
# action execution, and host-side grading. dockyard_rl exposes the seam where
# the two supported stacks diverge — action execution and observation source —
# behind this ABC, so the official OSWorld DesktopEnv and the in-house
# control-center stack are selectable implementations. Provisioning/setup/grading
# is backend-specific but reached through one start/observe/act/evaluate/reset/stop
# surface.

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from dockyard_rl.experience.cua.actions.pyautogui_parse import ParsedAction


@dataclass
class Observation:
    """One desktop observation handed to the policy.

    screenshot:         raw PNG bytes of the current screen, or None if the
                        backend produced no frame this turn.
    accessibility_tree: AT-SPI/pyatspi a11y XML, or None when the backend/image
                        does not expose one (screenshot-only is valid).
    terminal:           captured terminal text, or None.
    instruction:        the task instruction (constant across a task's turns).
    """

    screenshot: Optional[bytes]
    accessibility_tree: Optional[str]
    terminal: Optional[str]
    instruction: str


class OSWorldBackend(abc.ABC):
    """Pluggable provisioning/observation/actuation backend for OSWorld tasks.

    A backend owns the lifecycle of a single desktop episode. ``start`` provisions
    and resets a desktop for one task and returns an opaque handle; that handle is
    threaded into every other call. Implementations must tolerate concurrent
    episodes (one handle each) when driven from the async rollout path.
    """

    @abc.abstractmethod
    def start(self, task: dict[str, Any]) -> Any:
        """Provision/reset a desktop for ``task`` (an OSWorld task_config dict).

        Returns an opaque per-episode handle passed to the other methods.
        """

    @abc.abstractmethod
    def observe(self, handle: Any) -> Observation:
        """Return the current observation for the episode."""

    @abc.abstractmethod
    def act(self, handle: Any, action: "ParsedAction") -> None:
        """Execute one parsed action against the desktop.

        Code actions run their pyautogui body; control tokens are forwarded to
        the backend as WAIT/FAIL/DONE so they land in the backend's action
        history (OSWorld grading inspects the final action — e.g. an "infeasible"
        task is solved precisely by a trailing FAIL). The environment, not the
        backend, decides episode termination and calls ``evaluate``.
        """

    @abc.abstractmethod
    def evaluate(self, handle: Any) -> float:
        """Run host-side grading; return a reward in [0, 1]."""

    @abc.abstractmethod
    def reset(self, handle: Any, task: dict[str, Any]) -> Observation:
        """Reset an existing handle to a fresh state for ``task`` and return the
        first observation. Backends that cannot reset in place may stop and
        re-provision internally."""

    @abc.abstractmethod
    def stop(self, handle: Any) -> None:
        """Tear down the episode and release all resources for ``handle``."""
