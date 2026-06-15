# Duck-typed environment the vendored getters/dispatch require, decoupled from
# the upstream DesktopEnv. The getters reach the OSWorld guest server (:5000)
# either by direct HTTP (env.vm_ip/env.server_port) or through env.controller;
# both the official and control-center backends construct a GradingEnv pointed
# at whichever host runs the guest server.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GuestServerController(Protocol):
    """Subset of the OSWorld PythonController the vendored getters call.

    Implemented by clients/guest_server.py against the guest server (:5000).
    All methods mirror upstream desktop_env.controllers.python.PythonController
    semantics so the vendored getters behave identically.
    """

    def get_file(self, path: str) -> Any: ...
    def get_terminal_output(self) -> Any: ...
    def get_vm_directory_tree(self, path: str) -> Any: ...
    def get_vm_screen_size(self) -> Any: ...
    def get_vm_window_size(self, app_class_name: str) -> Any: ...
    def get_vm_wallpaper(self) -> Any: ...
    def get_accessibility_tree(self) -> Any: ...
    def execute_python_command(self, command: str) -> Any: ...


@dataclass
class GradingEnv:
    """Minimal env object threaded into the vendored getters and dispatch.

    vm_ip / server_port: host:port of the OSWorld guest server (the VM IP for the
        official docker provider; the control-center container's IP under the
        in-house backend). Getters that hit the guest server directly use these.
    cache_dir: local directory where cloud/vm files are downloaded for grading.
    controller: the GuestServerController bridging the remaining getter calls.
    action_history: ordered actions taken this episode; the dispatch inspects the
        last entry for FAIL (infeasible tasks score 1 on a trailing FAIL; normal
        tasks score 0 on one).
    """

    vm_ip: str
    server_port: int
    cache_dir: str
    controller: GuestServerController
    action_history: list[Any] = field(default_factory=list)
