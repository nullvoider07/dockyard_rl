# In-house control-center OSWorld backend (Phase 2).
#
# Same OSWorldBackend surface as the official DesktopEnv backend, but assembled
# from the user's stack against one provisioned ubuntu-desktop-code container:
#   observe  -> The-Eye  GET :8080/snapshot.png            (vision.py)
#   act      -> control-center ExecuteCommand :50051       (clients/control_center)
#               via pyautogui -> xdotool                   (actions/cc_transpile)
#   setup    -> OSWorld guest server :5000 SetupController  (clients/guest_server)
#   evaluate -> vendored evaluator dispatch + :5000 getters (vendor/desktop_env_eval)
#
# Provisioning is "attach": the backend connects to an already-running container
# at the configured host/ports. Ephemeral per-episode containers / a warm pool is
# an orchestration concern (Phase 3); `provision: docker` is reserved for it and
# rejected here rather than shipping an untested host-specific launcher. reset
# re-runs the task's setup steps on the attached container.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from dockyard_rl.experience.cua.actions.cc_transpile import pyautogui_to_xdotool
from dockyard_rl.experience.cua.actions.pyautogui_parse import ParsedAction
from dockyard_rl.experience.cua.backends.interface import Observation, OSWorldBackend
from dockyard_rl.experience.cua.clients.control_center import ControlCenterClient
from dockyard_rl.experience.cua.clients.guest_server import (
    GuestServerController,
    SetupController,
)
from dockyard_rl.experience.cua.provisioner import (
    HostLease,
    Provisioner,
    build_provisioner,
)
from dockyard_rl.experience.cua.vendor.desktop_env_eval import GradingEnv, evaluate
from dockyard_rl.experience.cua.vision import TheEyeClient

logger = logging.getLogger("dockyard_rl.cua.backends.control_center")


@dataclass
class _ControlCenterHandle:
    """Per-episode state for the control-center backend."""

    host: str
    lease: HostLease
    eye: TheEyeClient
    cc: ControlCenterClient
    guest: GuestServerController
    setup: SetupController
    task_config: dict[str, Any]
    guest_port: int
    cache_dir: str
    screen_size: tuple[int, int]
    desktop_size: tuple[int, int]
    display: str
    include_a11y: bool
    action_history: list[Any] = field(default_factory=list)


class ControlCenterBackend(OSWorldBackend):
    """OSWorldBackend over the in-house control-center / The-Eye / guest-server stack.

    Config keys (env.osworld; all optional unless noted):
        host:               container address the services run on (required).
        eye_port/eye_base_url/eye_token:   The-Eye (default :8080).
        cc_port/cc_use_ssl/cc_token/cc_jwt_secret:  control-center (default :50051).
        guest_port:         OSWorld guest server (default 5000).
        screen_width/height:        resolution advertised to the model.
        desktop_width/height:       physical desktop resolution (default = screen_*;
                                    used to scale the model's coordinates).
        display:            X display for xdotool (default ":0").
        cache_dir:          local dir for downloaded grading/setup files.
        include_a11y_text:  also fetch the a11y tree per observation (default False).
        client_password:    guest sudo password for setup steps (default "").
        provision:          host pool mode — "attach" (single host, default),
                            "static_pool" (cfg['hosts'] list), or "docker_pool"
                            (cfg['docker_image'] + pool_size). See provisioner.py.
    """

    def __init__(self, cfg: Optional[dict[str, Any]] = None) -> None:
        cfg = cfg or {}
        self._cfg = cfg
        # The provisioner (host pool) is built lazily on first start so a docker
        # pool's containers boot only when episodes actually run.
        self._provisioner: Optional[Provisioner] = None
        self.eye_port = int(cfg.get("eye_port", 8080))
        self.eye_base_url = cfg.get("eye_base_url")
        self.eye_token = cfg.get("eye_token")
        self.cc_port = int(cfg.get("cc_port", 50051))
        self.cc_use_ssl = bool(cfg.get("cc_use_ssl", False))
        self.cc_token = cfg.get("cc_token")
        self.cc_jwt_secret = cfg.get("cc_jwt_secret")
        self.guest_port = int(cfg.get("guest_port", 5000))
        self.screen_width = int(cfg.get("screen_width", 1920))
        self.screen_height = int(cfg.get("screen_height", 1080))
        self.desktop_width = int(cfg.get("desktop_width", self.screen_width))
        self.desktop_height = int(cfg.get("desktop_height", self.screen_height))
        self.display = cfg.get("display", ":0")
        self.cache_dir = cfg.get("cache_dir", "cache/osworld")
        self.include_a11y = bool(cfg.get("include_a11y_text", False))
        self.client_password = cfg.get("client_password", "")

    def _get_provisioner(self) -> Provisioner:
        if self._provisioner is None:
            self._provisioner = build_provisioner(self._cfg)
        return self._provisioner

    def start(self, task: dict[str, Any]) -> _ControlCenterHandle:
        lease = self._get_provisioner().acquire(task)
        host = lease.host
        eye = TheEyeClient(
            self.eye_base_url or f"http://{host}:{self.eye_port}",
            api_token=self.eye_token,
        )
        cc = ControlCenterClient(
            host,
            port=self.cc_port,
            use_ssl=self.cc_use_ssl,
            token=self.cc_token,
            jwt_secret=self.cc_jwt_secret,
        )
        cc.connect()
        guest = GuestServerController(host, server_port=self.guest_port)
        setup = SetupController(
            host,
            server_port=self.guest_port,
            cache_dir=self.cache_dir,
            client_password=self.client_password,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
        )
        handle = _ControlCenterHandle(
            host=host,
            lease=lease,
            eye=eye,
            cc=cc,
            guest=guest,
            setup=setup,
            task_config=task,
            guest_port=self.guest_port,
            cache_dir=self.cache_dir,
            screen_size=(self.screen_width, self.screen_height),
            desktop_size=(self.desktop_width, self.desktop_height),
            display=self.display,
            include_a11y=self.include_a11y,
        )
        # Run the task's setup steps (download/open/…) before the first obs.
        setup.setup(task.get("config", []) or [])
        return handle

    def observe(self, handle: _ControlCenterHandle) -> Observation:
        screenshot: Optional[bytes]
        try:
            screenshot = handle.eye.snapshot()
        except Exception as e:  # noqa: BLE001 - a dropped frame must not abort the turn
            logger.error("The-Eye snapshot failed: %s", e)
            screenshot = None
        a11y = handle.guest.get_accessibility_tree() if handle.include_a11y else None
        return Observation(
            screenshot=screenshot,
            accessibility_tree=a11y,
            terminal=None,
            instruction=handle.task_config.get("instruction", ""),
        )

    def act(self, handle: _ControlCenterHandle, action: ParsedAction) -> None:
        # Precedence mirrors the official backend: FAIL must land last in the
        # action history (grading inspects the final action); else execute code;
        # else record the bare control token; a no-op turn does nothing.
        if action.control == "fail":
            handle.action_history.append("FAIL")
            return
        if action.code:
            commands = pyautogui_to_xdotool(
                action.code,
                screen_size=handle.screen_size,
                target_size=handle.desktop_size,
                display=handle.display,
            )
            for cmd in commands:
                result = handle.cc.execute(cmd)
                if not result.get("success", False):
                    logger.warning("control-center command failed: %s", result.get("message"))
            handle.action_history.append(action.code)
            return
        if action.control == "wait":
            handle.action_history.append("WAIT")
            return
        if action.control == "done":
            handle.action_history.append("DONE")
            return
        # No code and no control token: nothing to actuate.

    def evaluate(self, handle: _ControlCenterHandle) -> float:
        grading_env = GradingEnv(
            vm_ip=handle.host,
            server_port=handle.guest_port,
            cache_dir=handle.cache_dir,
            controller=handle.guest,
            action_history=handle.action_history,
        )

        def _setup_wrapper(config: list[dict[str, Any]]) -> None:
            handle.setup.setup(config)

        return float(
            evaluate(handle.task_config, grading_env, setup_fn=_setup_wrapper)
        )

    def reset(self, handle: _ControlCenterHandle, task: dict[str, Any]) -> Observation:
        # Attach mode: re-run the new task's setup on the same container and clear
        # the action history. (A pooled/ephemeral provisioner would re-provision.)
        handle.task_config = task
        handle.action_history.clear()
        handle.setup.setup(task.get("config", []) or [])
        return self.observe(handle)

    def stop(self, handle: _ControlCenterHandle) -> None:
        try:
            handle.cc.close()
        except Exception:  # noqa: BLE001 - teardown must not raise into the batch
            pass
        # Return the host to the pool for the next episode.
        self._get_provisioner().release(handle.lease)
