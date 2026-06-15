# Clients for the OSWorld guest server (:5000), the evaluator/setup bridge for
# the control-center backend (Phase 2, option b: the guest server is installed
# into the ubuntu-desktop-code image). Faithful to the pinned upstream
# desktop_env.controllers.python.PythonController +
# desktop_env.controllers.setup.SetupController
# (xlang-ai/OSWorld 705623ca18e0055dd995fd5a350d6588cff2caf5): same HTTP
# endpoints, payloads, and semantics, so the vendored getters/setup behave
# identically. Deviations: file upload uses plain requests multipart instead of
# requests_toolbelt.MultipartEncoder; only the setup types the current task
# domains (os, libreoffice_calc) use are implemented (execute/sleep/download/
# open/activate_window), with a clear error for the rest.

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger("dockyard_rl.cua.clients.guest_server")

# pyautogui import preamble the guest wraps around python commands, copied
# verbatim from the pinned PythonController so execute_python_command behaves
# identically (the guest image provides pyautogui).
PYAUTOGUI_PKGS_PREFIX = (
    "import pyautogui; import time; import platform; "
    "pyautogui.FAILSAFE = False; "
    "_osworld_shift_chars = '~!@#$%^&*()_+' + chr(123) + chr(125) + '|:\"<>?'; "
    "_osworld_linux_shift_chars = '~!@#$%^&*()_+' + chr(123) + chr(125) + '|:\">?'; "
    "pyautogui.isShiftCharacter = lambda character: character.isupper() or "
    "character in (_osworld_linux_shift_chars if platform.system() == 'Linux' "
    "else _osworld_shift_chars); "
    "{command}"
)

DEFAULT_SERVER_PORT = 5000


class GuestServerController:
    """HTTP client for the guest server's observation/state endpoints (:5000).

    Implements the GuestServerController Protocol the vendored evaluator getters
    require (see vendor/desktop_env_eval/grading_env.py). Constructed by the
    backend pointed at the host running the guest server and stored on
    GradingEnv.controller.
    """

    def __init__(
        self,
        vm_ip: str,
        server_port: int = DEFAULT_SERVER_PORT,
        pkgs_prefix: str = PYAUTOGUI_PKGS_PREFIX,
        retry_times: int = 3,
        retry_interval: float = 5.0,
    ) -> None:
        self.vm_ip = vm_ip
        self.server_port = server_port
        self.http_server = f"http://{vm_ip}:{server_port}"
        self.pkgs_prefix = pkgs_prefix
        self.retry_times = retry_times
        self.retry_interval = retry_interval

    def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> Optional[requests.Response]:
        """Issue a request to the guest server, retrying on error/non-200."""
        url = self.http_server + path
        for _ in range(self.retry_times):
            try:
                resp = requests.request(method, url, **kwargs)
                if resp.status_code == 200:
                    return resp
                logger.error("guest-server %s %s: status %d", method, path, resp.status_code)
            except Exception as e:  # noqa: BLE001 - retry on any transport error
                logger.error("guest-server %s %s error: %s", method, path, e)
            time.sleep(self.retry_interval)
        logger.error("guest-server %s %s failed after %d tries", method, path, self.retry_times)
        return None

    def get_accessibility_tree(self) -> Optional[str]:
        resp = self._request("GET", "/accessibility")
        return resp.json()["AT"] if resp is not None else None

    def get_terminal_output(self) -> Optional[str]:
        resp = self._request("GET", "/terminal")
        return resp.json()["output"] if resp is not None else None

    def get_file(self, path: str) -> Optional[bytes]:
        resp = self._request("POST", "/file", data={"file_path": path})
        return resp.content if resp is not None else None

    def execute_python_command(self, command: str) -> Optional[Dict[str, Any]]:
        command_list = ["python", "-c", self.pkgs_prefix.format(command=command)]
        resp = self._request(
            "POST",
            "/execute",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"command": command_list, "shell": False}),
            timeout=120,
        )
        return resp.json() if resp is not None else None

    def get_vm_screen_size(self) -> Optional[Any]:
        resp = self._request("POST", "/screen_size")
        return resp.json() if resp is not None else None

    def get_vm_window_size(self, app_class_name: str) -> Optional[Any]:
        resp = self._request("POST", "/window_size", data={"app_class_name": app_class_name})
        return resp.json() if resp is not None else None

    def get_vm_wallpaper(self) -> Optional[bytes]:
        resp = self._request("POST", "/wallpaper")
        return resp.content if resp is not None else None

    def get_vm_directory_tree(self, path: str) -> Optional[Dict[str, Any]]:
        resp = self._request(
            "POST",
            "/list_directory",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"path": path}),
        )
        return resp.json()["directory_tree"] if resp is not None else None


# Setup step types implemented for the current task domains (os, libreoffice_calc).
# Add others (chrome_open_tabs, googledrive, launch, …) per-domain in later slices.
_SUPPORTED_SETUP_TYPES = ("execute", "sleep", "download", "open", "activate_window")


class SetupController:
    """Runs OSWorld task ``config[]`` / evaluator ``postconfig`` setup steps
    against the guest server (:5000), faithful to the pinned SetupController.

    Only the subset of setup types used by the os/libreoffice_calc domains is
    implemented; an unsupported type raises NotImplementedError naming it, so
    extending coverage is explicit.
    """

    def __init__(
        self,
        vm_ip: str,
        server_port: int = DEFAULT_SERVER_PORT,
        cache_dir: str = "cache",
        client_password: str = "",
        screen_width: int = 1920,
        screen_height: int = 1080,
        connect_retries: int = 3,
        connect_interval: float = 5.0,
    ) -> None:
        self.vm_ip = vm_ip
        self.server_port = server_port
        self.http_server = f"http://{vm_ip}:{server_port}"
        self.cache_dir = cache_dir
        self.client_password = client_password
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.connect_retries = connect_retries
        self.connect_interval = connect_interval

    def setup(self, config: List[Dict[str, Any]]) -> bool:
        """Run an ordered list of setup steps; raise on the first failure."""
        if not config:
            return True
        # Block until the guest server is reachable.
        for attempt in range(self.connect_retries):
            try:
                requests.get(self.http_server + "/terminal", timeout=10)
                break
            except Exception:  # noqa: BLE001 - guest may still be booting
                if attempt == self.connect_retries - 1:
                    logger.error("guest server %s unreachable", self.http_server)
                    return False
                time.sleep(self.connect_interval)

        for i, cfg in enumerate(config):
            cfg_type = cfg["type"]
            params = cfg.get("parameters", {})
            handler = getattr(self, f"_{cfg_type}_setup", None)
            if handler is None:
                raise NotImplementedError(
                    f"setup type {cfg_type!r} is not implemented in this build "
                    f"(supported: {', '.join(_SUPPORTED_SETUP_TYPES)}); add it "
                    f"when its task domain is brought online."
                )
            logger.info("setup step %d/%d: %s", i + 1, len(config), cfg_type)
            handler(**params)
        return True

    def _replace_screen_env(self, command: Union[str, List[str]]) -> Union[str, List[str]]:
        repl = {
            "{CLIENT_PASSWORD}": self.client_password,
            "{SCREEN_WIDTH_HALF}": str(self.screen_width // 2),
            "{SCREEN_HEIGHT_HALF}": str(self.screen_height // 2),
            "{SCREEN_WIDTH}": str(self.screen_width),
            "{SCREEN_HEIGHT}": str(self.screen_height),
        }

        def sub(s: str) -> str:
            for k, v in repl.items():
                s = s.replace(k, v)
            return s

        if isinstance(command, str):
            return sub(command)
        return [sub(item) for item in command]

    def _execute_setup(
        self,
        command: List[str],
        stdout: str = "",
        stderr: str = "",
        shell: bool = False,
        until: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not command:
            raise Exception("Empty command to launch.")
        command = self._replace_screen_env(command)  # type: ignore[assignment]
        payload = json.dumps({"command": command, "shell": shell})
        until = until or {}
        nb_failings = 0
        os.makedirs(self.cache_dir, exist_ok=True)

        while True:
            results: Any = None
            try:
                resp = requests.post(
                    self.http_server + "/setup/execute",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                )
                if resp.status_code == 200:
                    results = resp.json()
                    if stdout:
                        with open(os.path.join(self.cache_dir, stdout), "w") as f:
                            f.write(results["output"])
                    if stderr:
                        with open(os.path.join(self.cache_dir, stderr), "w") as f:
                            f.write(results["error"])
                else:
                    logger.error("setup execute failed: %s", resp.text)
                    nb_failings += 1
            except requests.exceptions.RequestException as e:
                logger.error("setup execute request error: %s", e)
                nb_failings += 1

            if not until:
                terminates = True
            elif results is not None:
                terminates = (
                    ("returncode" in until and results.get("returncode") == until["returncode"])
                    or ("stdout" in until and until["stdout"] in results.get("output", ""))
                    or ("stderr" in until and until["stderr"] in results.get("error", ""))
                )
            else:
                terminates = False
            if terminates or nb_failings >= 5:
                break
            time.sleep(0.3)

    def _sleep_setup(self, seconds: float) -> None:
        time.sleep(seconds)

    def _download_setup(self, files: List[Dict[str, str]]) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        for f in files:
            url = f["url"]
            path = f["path"]
            if not url or not path:
                raise Exception(f"Setup Download - invalid url ({url}) or path ({path}).")
            cache_path = os.path.join(
                self.cache_dir,
                "{:}_{:}".format(uuid.uuid5(uuid.NAMESPACE_URL, url), os.path.basename(path)),
            )
            if not os.path.exists(cache_path):
                downloaded = False
                for attempt in range(3):
                    try:
                        resp = requests.get(url, stream=True, timeout=300)
                        resp.raise_for_status()
                        with open(cache_path, "wb") as out:
                            for chunk in resp.iter_content(chunk_size=8192):
                                if chunk:
                                    out.write(chunk)
                        downloaded = True
                        break
                    except requests.RequestException as e:
                        logger.error("download %s failed (%d left): %s", url, 2 - attempt, e)
                        if os.path.exists(cache_path):
                            os.remove(cache_path)
                if not downloaded:
                    raise requests.RequestException(f"Failed to download {url}.")

            with open(cache_path, "rb") as fh:
                resp = requests.post(
                    self.http_server + "/setup/upload",
                    data={"file_path": path},
                    files={"file_data": (os.path.basename(path), fh)},
                    timeout=600,
                )
            if resp.status_code != 200:
                raise requests.RequestException(
                    f"Upload of {path} failed: status {resp.status_code}"
                )

    def _open_setup(self, path: str) -> None:
        if not path:
            raise Exception(f"Setup Open - invalid path ({path}).")
        resp = requests.post(
            self.http_server + "/setup/open_file",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"path": path}),
            timeout=1810,
        )
        resp.raise_for_status()

    def _activate_window_setup(
        self, window_name: str, strict: bool = False, by_class: bool = False
    ) -> None:
        if not window_name:
            raise Exception(f"Setup Activate Window - invalid name ({window_name}).")
        resp = requests.post(
            self.http_server + "/setup/activate_window",
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {"window_name": window_name, "strict": strict, "by_class": by_class}
            ),
        )
        if resp.status_code != 200:
            logger.error("activate_window %s failed: %s", window_name, resp.text)
