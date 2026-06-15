# macOS CUA backend (Phase 4). Drives a macOS guest (qcow2 in a KVM-in-container,
# e.g. Environments/mac15-base / OSX-KVM) where control-center and The-Eye are
# baked into the guest image:
#   observe  -> The-Eye  GET /snapshot.png (:8080)
#   act      -> control-center ExecuteCommand :50051, via pyautogui -> cliclick /
#               osascript shell strings (cc_transpile.pyautogui_to_macos); the
#               agent runs cliclick directly and osascript/compound via sh -c
#   provision-> the same warm-pool provisioner as the Linux/Windows backends, with
#               the macOS container image / host pool (qcow2-overlay reset).
#
# Like the Windows backend, grading is custom-only: the OSWorld suite is
# Ubuntu-authored and grades through the :5000 guest server, absent on the macOS
# guest. macOS is infrastructure for *custom* task sets; their evaluators come
# from experience/cua/grading (set env.osworld.grader). evaluate therefore raises
# unless a grader is configured.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from dockyard_rl.experience.cua.actions.cc_transpile import pyautogui_to_macos
from dockyard_rl.experience.cua.actions.pyautogui_parse import ParsedAction
from dockyard_rl.experience.cua.backends.interface import Observation, OSWorldBackend
from dockyard_rl.experience.cua.clients.control_center import ControlCenterClient
from dockyard_rl.experience.cua.grading import Grader, build_grader
from dockyard_rl.experience.cua.provisioner import (
    HostLease,
    Provisioner,
    build_provisioner,
)
from dockyard_rl.experience.cua.vision import TheEyeClient

logger = logging.getLogger("dockyard_rl.cua.backends.macos")


@dataclass
class _MacHandle:
    host: str
    lease: HostLease
    eye: TheEyeClient
    cc: ControlCenterClient
    task_config: dict[str, Any]
    screen_size: tuple[int, int]
    desktop_size: tuple[int, int]
    action_history: list[Any] = field(default_factory=list)


class MacBackend(OSWorldBackend):
    """OSWorldBackend for a macOS guest over control-center + The-Eye.

    Config keys (env.osworld; mirrors the windows backend with macOS defaults):
        provision/host/hosts/docker_image/pool_size/...:  host pool (provisioner.py).
        eye_port/eye_base_url/eye_token:   The-Eye (default :8080).
        cc_port/cc_use_ssl/cc_token/cc_jwt_secret:  control-center (default :50051).
        screen_width/height, desktop_width/height:  coord spaces (scaling).
        grader/evaluator_modules/judge:  custom-task grading (experience/cua/grading);
                 absent grader → evaluate raises.
    """

    def __init__(self, cfg: Optional[dict[str, Any]] = None) -> None:
        cfg = cfg or {}
        self._cfg = cfg
        self._provisioner: Optional[Provisioner] = None
        self.eye_port = int(cfg.get("eye_port", 8080))
        self.eye_base_url = cfg.get("eye_base_url")
        self.eye_token = cfg.get("eye_token")
        self.cc_port = int(cfg.get("cc_port", 50051))
        self.cc_use_ssl = bool(cfg.get("cc_use_ssl", False))
        self.cc_token = cfg.get("cc_token")
        self.cc_jwt_secret = cfg.get("cc_jwt_secret")
        self.screen_width = int(cfg.get("screen_width", 1920))
        self.screen_height = int(cfg.get("screen_height", 1080))
        self.desktop_width = int(cfg.get("desktop_width", self.screen_width))
        self.desktop_height = int(cfg.get("desktop_height", self.screen_height))
        self._grader: Optional[Grader] = build_grader(cfg)

    def _get_provisioner(self) -> Provisioner:
        if self._provisioner is None:
            self._provisioner = build_provisioner(self._cfg)
        return self._provisioner

    def start(self, task: dict[str, Any]) -> _MacHandle:
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
        return _MacHandle(
            host=host,
            lease=lease,
            eye=eye,
            cc=cc,
            task_config=task,
            screen_size=(self.screen_width, self.screen_height),
            desktop_size=(self.desktop_width, self.desktop_height),
        )

    def observe(self, handle: _MacHandle) -> Observation:
        try:
            screenshot: Optional[bytes] = handle.eye.snapshot()
        except Exception as e:  # noqa: BLE001 - a dropped frame must not abort the turn
            logger.error("The-Eye snapshot failed: %s", e)
            screenshot = None
        return Observation(
            screenshot=screenshot,
            accessibility_tree=None,  # macOS a11y is not wired (screenshot-primary)
            terminal=None,
            instruction=handle.task_config.get("instruction", ""),
        )

    def act(self, handle: _MacHandle, action: ParsedAction) -> None:
        # Same precedence as the other backends: FAIL last (grading inspects the
        # final action), else execute code, else record the control token.
        if action.control == "fail":
            handle.action_history.append("FAIL")
            return
        if action.code:
            for cmd in pyautogui_to_macos(
                action.code,
                screen_size=handle.screen_size,
                target_size=handle.desktop_size,
            ):
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

    def evaluate(self, handle: _MacHandle) -> float:
        if self._grader is not None:
            return float(self._grader(handle.task_config, handle))
        raise NotImplementedError(
            "MacBackend has no grading configured: OSWorld's evaluators are "
            "Ubuntu-authored and run against the :5000 guest server, absent on the "
            "macOS guest. Set env.osworld.grader (e.g. 'custom') and author "
            "evaluators in experience/cua/grading for macOS task sets."
        )

    def reset(self, handle: _MacHandle, task: dict[str, Any]) -> Observation:
        handle.task_config = task
        handle.action_history.clear()
        return self.observe(handle)

    def stop(self, handle: _MacHandle) -> None:
        try:
            handle.cc.close()
        except Exception:  # noqa: BLE001 - teardown must not raise into the batch
            pass
        self._get_provisioner().release(handle.lease)
