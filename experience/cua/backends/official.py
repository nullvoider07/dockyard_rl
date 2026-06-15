# Official OSWorld backend: wraps upstream desktop_env.DesktopEnv.
#
# desktop_env is a build-time IMAGE dependency (pinned xlang-ai/OSWorld
# 705623ca18e0055dd995fd5a350d6588cff2caf5), not vendored — the import is guarded
# so this module loads on hosts without it (offline validation, the trainer
# fleet). One DesktopEnv == one VM == one episode: ``start`` constructs+resets a
# DesktopEnv for the task (the docker KVM provider has no snapshot, so reset is a
# boot+setup), and ``stop`` closes it. DesktopEnv.evaluate() is the host-side
# reward; WAIT/FAIL/DONE are forwarded into step() so OSWorld's action-history
# grading (an "infeasible" task is solved by a trailing FAIL) is correct.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from dockyard_rl.experience.cua.actions.pyautogui_parse import ParsedAction
from dockyard_rl.experience.cua.backends.interface import Observation, OSWorldBackend

try:
    from desktop_env.desktop_env import DesktopEnv  # type: ignore[import]
except ImportError:  # pragma: no cover - desktop_env is an image dependency
    DesktopEnv = None  # type: ignore[assignment, misc]


@dataclass
class _OfficialHandle:
    """Per-episode state for the official backend."""

    env: Any  # desktop_env.DesktopEnv
    last_obs: dict[str, Any]
    task_config: dict[str, Any]


class OfficialDesktopEnvBackend(OSWorldBackend):
    """OSWorldBackend backed by upstream ``desktop_env.DesktopEnv``.

    Config keys (all optional; OSWorld defaults applied):
        provider_name:      virtualization provider (default "docker" — KVM-in-
                            container, the dependency-free local path).
        region/path_to_vm/snapshot_name: provider-specific knobs forwarded as-is.
        screen_width/screen_height: desktop resolution (default 1920x1080).
        headless:           run the VM headless (default True).
        require_a11y_tree:  request the accessibility tree in observations
                            (default True).
        require_terminal:   request terminal output in observations (default
                            False).
        client_password:    guest sudo password (provider default applied when "").
        cache_dir:          DesktopEnv cache directory (default "cache").
        pause_after_action: seconds DesktopEnv.step waits after an action; a task
                            row's ``pause_after_action`` overrides per-episode.
    """

    def __init__(self, cfg: Optional[dict[str, Any]] = None) -> None:
        cfg = cfg or {}
        self.provider_name = cfg.get("provider_name", "docker")
        self.region = cfg.get("region")
        self.path_to_vm = cfg.get("path_to_vm")
        self.snapshot_name = cfg.get("snapshot_name", "init_state")
        self.screen_width = int(cfg.get("screen_width", 1920))
        self.screen_height = int(cfg.get("screen_height", 1080))
        self.headless = bool(cfg.get("headless", True))
        self.require_a11y_tree = bool(cfg.get("require_a11y_tree", True))
        self.require_terminal = bool(cfg.get("require_terminal", False))
        self.client_password = cfg.get("client_password", "")
        self.cache_dir = cfg.get("cache_dir", "cache")
        self.default_pause = float(cfg.get("pause_after_action", 2.0))

    def _require_desktop_env(self) -> None:
        if DesktopEnv is None:
            raise RuntimeError(
                "desktop_env is not importable. The official OSWorld backend "
                "needs the upstream desktop_env package installed in the image "
                "running this environment (it is a build-time dependency, not "
                "vendored)."
            )

    def start(self, task: dict[str, Any]) -> _OfficialHandle:
        self._require_desktop_env()
        assert DesktopEnv is not None  # narrowed by _require_desktop_env
        env = DesktopEnv(
            provider_name=self.provider_name,
            region=self.region,
            path_to_vm=self.path_to_vm,
            snapshot_name=self.snapshot_name,
            action_space="pyautogui",
            cache_dir=self.cache_dir,
            screen_size=(self.screen_width, self.screen_height),
            headless=self.headless,
            require_a11y_tree=self.require_a11y_tree,
            require_terminal=self.require_terminal,
            os_type="Ubuntu",
            client_password=self.client_password,
        )
        obs = env.reset(task_config=task)
        return _OfficialHandle(env=env, last_obs=obs or {}, task_config=task)

    def observe(self, handle: _OfficialHandle) -> Observation:
        obs = handle.last_obs or {}
        return Observation(
            screenshot=obs.get("screenshot"),
            accessibility_tree=obs.get("accessibility_tree"),
            terminal=obs.get("terminal"),
            instruction=obs.get("instruction") or handle.task_config.get("instruction", ""),
        )

    def act(self, handle: _OfficialHandle, action: ParsedAction) -> None:
        pause = float(handle.task_config.get("pause_after_action", self.default_pause))
        # Precedence: FAIL must reach action history (grading inspects the last
        # action); otherwise run the code; otherwise the bare control token. A
        # no-op turn (no code, no token) does not step — the environment reprompts
        # and the screen is unchanged.
        if action.control == "fail":
            step_action: Any = "FAIL"
        elif action.code:
            step_action = action.code
        elif action.control == "wait":
            step_action = "WAIT"
        elif action.control == "done":
            step_action = "DONE"
        else:
            return
        obs, _reward, _done, _info = handle.env.step(step_action, pause=pause)
        if obs:
            handle.last_obs = obs

    def evaluate(self, handle: _OfficialHandle) -> float:
        return float(handle.env.evaluate())

    def reset(self, handle: _OfficialHandle, task: dict[str, Any]) -> Observation:
        obs = handle.env.reset(task_config=task)
        handle.last_obs = obs or {}
        handle.task_config = task
        return self.observe(handle)

    def stop(self, handle: _OfficialHandle) -> None:
        try:
            handle.env.close()
        except Exception:  # noqa: BLE001 - teardown must not raise into the batch
            pass
