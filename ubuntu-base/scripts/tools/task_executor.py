"""
task_executor_ubuntu.py — REST API task executor for the Ubuntu 24.04 AI Agent environment.
"""

import base64
import difflib
import hashlib
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from http import HTTPStatus

from flask import Flask, jsonify, request
from waitress import serve

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TASK_BASE_DIR = os.environ.get("TASK_BASE_DIR", "/workspace/tasks")
API_PORT      = int(os.environ.get("API_PORT", "9090"))
API_TOKEN     = os.environ.get("API_TOKEN", "")
# Concurrent multi-turn sessions each hold a waitress worker thread for the
# duration of an exec/finish, so the pool is larger than the single-shot default.
_API_THREADS  = int(os.environ.get("DOCKYARD_TASK_EXECUTOR_THREADS", "32"))

# ---------------------------------------------------------------------------
# Logging  →  /workspace/tasks/task_executor.log
# ---------------------------------------------------------------------------
os.makedirs(TASK_BASE_DIR, exist_ok=True)
LOG_FILE = os.path.join(TASK_BASE_DIR, "task_executor.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("task_executor")

# ---------------------------------------------------------------------------
# In-memory task store
# ---------------------------------------------------------------------------
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()
_TASK_MAX_AGE = int(os.environ.get("TASK_MAX_AGE", "3600"))  # 1 hour default


def _evict_old_tasks() -> None:
    """Drop completed/failed tasks older than TASK_MAX_AGE seconds."""
    cutoff = time.monotonic() - _TASK_MAX_AGE
    with _tasks_lock:
        stale = [
            tid for tid, t in _tasks.items()
            if t["status"] not in ("pending", "running")
            and t.get("_created", 0) < cutoff
        ]
        for tid in stale:
            _tasks.pop(tid)

app = Flask(__name__)

def _check_auth() -> bool:
    """Return True if the request is authorised (or auth is disabled)."""
    if not API_TOKEN:
        return True  # auth disabled if no token configured
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {API_TOKEN}"

# ===========================================================================
# Custom exceptions
# ===========================================================================

class _TaskTimeoutError(RuntimeError):
    """Raised by _run() when a subprocess exceeds its allotted time."""


# ===========================================================================
# Subprocess helper
# ===========================================================================

def _run(
    command: list[str] | str,
    cwd: str | None = None,
    timeout: int = 120,
    shell: bool = False,
) -> tuple[int, str, str]:
    """
    Run a command in a new process group so the entire child tree can be
    killed on timeout via os.killpg + SIGKILL (POSIX).

    • list[str]  →  all internal commands (git clone/checkout/apply/diff).
    • shell=True →  user-supplied test_command and lint_command only.

    Raises _TaskTimeoutError on timeout.
    Returns (exit_code, stdout, stderr).
    """
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise _TaskTimeoutError(f"Command timed out after {timeout}s")


# ===========================================================================
# Test result parsers
# ===========================================================================

def _parse_pytest(text: str) -> tuple[int, int]:
    passed, failed = 0, 0
    m = re.search(r"(\d+)\s+passed", text)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+)\s+failed", text)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+)\s+error", text)
    if m:
        failed += int(m.group(1))
    return passed, failed


def _parse_cargo(text: str) -> tuple[int, int]:
    passed, failed = 0, 0
    for m in re.finditer(r"test result:.*?(\d+)\s+passed;\s*(\d+)\s+failed", text):
        passed += int(m.group(1))
        failed += int(m.group(2))
    return passed, failed


def _parse_go(text: str) -> tuple[int, int]:
    passed = len(re.findall(r"^--- PASS:", text, re.MULTILINE))
    failed = len(re.findall(r"^--- FAIL:", text, re.MULTILINE))
    if passed == 0 and failed == 0:
        passed = len(re.findall(r"^ok\s+\S+",   text, re.MULTILINE))
        failed = len(re.findall(r"^FAIL\s+\S+", text, re.MULTILINE))
    return passed, failed


def _parse_jest(text: str) -> tuple[int, int]:
    passed, failed = 0, 0
    m = re.search(r"^Tests:\s+(.+)$", text, re.MULTILINE)
    if m:
        summary = m.group(1)
        p = re.search(r"(\d+)\s+passed", summary)
        f = re.search(r"(\d+)\s+failed", summary)
        if p:
            passed = int(p.group(1))
        if f:
            failed = int(f.group(1))
    return passed, failed


def _parse_dotnet(text: str) -> tuple[int, int]:
    passed, failed = 0, 0
    m = re.search(r"Failed:\s*(\d+),\s*Passed:\s*(\d+)", text)
    if m:
        failed = int(m.group(1))
        passed = int(m.group(2))
    return passed, failed


def _parse_junit(text: str) -> tuple[int, int]:
    passed_total = failed_total = 0
    for m in re.finditer(
        r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)", text
    ):
        run      = int(m.group(1))
        failures = int(m.group(2))
        errors   = int(m.group(3))
        failed_total += failures + errors
        passed_total += max(run - failures - errors, 0)
    return passed_total, failed_total


def _dispatch_test_parser(test_command: str, text: str) -> tuple[int, int]:
    cmd = test_command.lower()
    if "pytest" in cmd or "py.test" in cmd:
        return _parse_pytest(text)
    if "cargo" in cmd:
        return _parse_cargo(text)
    if "go test" in cmd:
        return _parse_go(text)
    if (
        "jest" in cmd
        or ("npm" in cmd and "test" in cmd)
        or ("yarn" in cmd and "test" in cmd)
        or ("pnpm" in cmd and "test" in cmd)
    ):
        return _parse_jest(text)
    if "dotnet" in cmd:
        return _parse_dotnet(text)
    if "mvn" in cmd or "gradle" in cmd or "sbt" in cmd or "junit" in cmd:
        return _parse_junit(text)
    for parser in (
        _parse_pytest, _parse_cargo, _parse_go,
        _parse_jest, _parse_dotnet, _parse_junit,
    ):
        p, f = parser(text)
        if p or f:
            return p, f
    return 0, 0


# ===========================================================================
# Lint error parser
# ===========================================================================

def _parse_lint_errors(lint_command: str, text: str, exit_code: int) -> int:
    """
    Extract an error count from linter output.
    Soft scoring only — never changes task status.
    """
    cmd = lint_command.lower()

    if "ruff" in cmd:
        m = re.search(r"Found\s+(\d+)\s+error", text)
        if m:
            return int(m.group(1))
        if "--output-format json" in cmd or "-o json" in cmd:
            try:
                import json
                return len(json.loads(text))
            except Exception:
                pass

    if "flake8" in cmd:
        return len([l for l in text.splitlines() if re.match(r".+:\d+:\d+:\s+[EWF]", l)])

    if "mypy" in cmd:
        m = re.search(r"Found\s+(\d+)\s+error", text)
        if m:
            return int(m.group(1))
        return text.count(": error:")

    if "pylint" in cmd:
        return len(re.findall(r"^\S+:\d+:\d+:\s+[EF]\d{4}:", text, re.MULTILINE))

    if "clippy" in cmd or ("cargo" in cmd and "check" in cmd):
        return len(re.findall(r"^error\[", text, re.MULTILINE))

    if "eslint" in cmd:
        if "--format json" in cmd or "-f json" in cmd:
            try:
                import json
                data = json.loads(text)
                return sum(
                    sum(1 for msg in f.get("messages", []) if msg.get("severity") == 2)
                    for f in data
                )
            except Exception:
                pass
        m = re.search(r"(\d+)\s+error", text)
        return int(m.group(1)) if m else 0

    if "go vet" in cmd or "staticcheck" in cmd:
        return len([l for l in text.splitlines() if l.strip()])

    if "clang-tidy" in cmd or "cppcheck" in cmd:
        return len(re.findall(r"\berror\b", text, re.IGNORECASE))

    if "dotnet" in cmd and "build" in cmd:
        m = re.search(r"(\d+)\s+Error\(s\)", text)
        return int(m.group(1)) if m else 0

    if exit_code != 0:
        return len(re.findall(r"\berror\b", text, re.IGNORECASE))
    return 0


# ===========================================================================
# Patch normaliser + similarity scorer
# ===========================================================================

def _normalise_patch(patch: str) -> list[str]:
    kept: list[str] = []
    for line in patch.splitlines():
        if (
            line.startswith("diff ")
            or line.startswith("index ")
            or line.startswith("--- ")
            or line.startswith("+++ ")
            or line.startswith("@@ ")
        ):
            continue
        kept.append(line)
    return kept


def _patch_similarity(agent_patch: str, reference_patch: str) -> float:
    a = _normalise_patch(agent_patch)
    b = _normalise_patch(reference_patch)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ===========================================================================
# Docker image-task support (per-instance prebuilt images)
# ===========================================================================
# Generic image-run mode: pull a prebuilt image on demand, stage a set of
# workspace files into a host dir mounted at /workspace, run an entry command,
# and read designated result files back. Benchmark-specific assembly (e.g. the
# SWE-bench Pro entryscript / log parsing) stays in the reward, not here.
#
# Image cache is bounded by an LRU policy: images pulled here are tracked by
# last-use time and evicted (least-recently-used first, never while in use)
# once the tracked footprint exceeds DOCKYARD_SANDBOX_IMAGE_CACHE_MAX_GB.

DOCKER_BIN              = os.environ.get("DOCKYARD_DOCKER_BIN", "docker")
_IMAGE_PULL_TIMEOUT     = int(os.environ.get("DOCKYARD_SANDBOX_IMAGE_PULL_TIMEOUT", "1800"))
_IMAGE_CACHE_MAX_BYTES  = int(
    float(os.environ.get("DOCKYARD_SANDBOX_IMAGE_CACHE_MAX_GB", "200")) * (1024 ** 3)
)

_image_lock = threading.Lock()
_image_last_used: dict[str, float] = {}
_image_in_use:    dict[str, int]   = {}
_docker_ready = False


def _docker_available() -> bool:
    """True once `docker info` succeeds; rechecks while the daemon is down."""
    global _docker_ready
    if _docker_ready:
        return True
    rc, _, _ = _run([DOCKER_BIN, "info"], timeout=30)
    _docker_ready = rc == 0
    return _docker_ready


def _image_present(ref: str) -> bool:
    rc, _, _ = _run([DOCKER_BIN, "image", "inspect", ref], timeout=60)
    return rc == 0


def _image_size_bytes(ref: str) -> int:
    rc, out, _ = _run(
        [DOCKER_BIN, "image", "inspect", "--format", "{{.Size}}", ref], timeout=60
    )
    out = out.strip()
    return int(out) if rc == 0 and out.isdigit() else 0


def _tracked_total_bytes() -> int:
    return sum(_image_size_bytes(img) for img in list(_image_last_used))


def _ensure_image(ref: str, timeout: int = _IMAGE_PULL_TIMEOUT) -> None:
    """Reserve `ref` and pull it if absent. Caller must _release_image later."""
    with _image_lock:
        _image_in_use[ref] = _image_in_use.get(ref, 0) + 1
        _image_last_used[ref] = time.monotonic()
    if _image_present(ref):
        return
    rc, _out, err = _run([DOCKER_BIN, "pull", ref], timeout=timeout)
    if rc != 0:
        with _image_lock:
            n = _image_in_use.get(ref, 1) - 1
            if n <= 0:
                _image_in_use.pop(ref, None)
            else:
                _image_in_use[ref] = n
        raise RuntimeError(f"docker pull {ref} failed (rc={rc}): {err.strip()[:500]}")


def _release_image(ref: str) -> None:
    """Drop one reference to `ref`, refresh its LRU stamp, and evict if over cap."""
    with _image_lock:
        n = _image_in_use.get(ref, 0) - 1
        if n <= 0:
            _image_in_use.pop(ref, None)
        else:
            _image_in_use[ref] = n
        _image_last_used[ref] = time.monotonic()
        _evict_lru_locked()


def _evict_lru_locked() -> None:
    """Evict tracked images (LRU first, never in-use) until under the cap.

    Called while holding `_image_lock`. Scoped to images this executor pulled
    so it never removes the base/sandbox images.
    """
    if _IMAGE_CACHE_MAX_BYTES <= 0:
        return
    total = _tracked_total_bytes()
    if total <= _IMAGE_CACHE_MAX_BYTES:
        return
    candidates = sorted(
        (img for img in _image_last_used if _image_in_use.get(img, 0) == 0),
        key=lambda i: _image_last_used[i],
    )
    for img in candidates:
        if total <= _IMAGE_CACHE_MAX_BYTES:
            break
        size = _image_size_bytes(img)
        rc, _, _ = _run([DOCKER_BIN, "image", "rm", "-f", img], timeout=120)
        if rc == 0:
            _image_last_used.pop(img, None)
            total -= size
            log.info("Evicted LRU image %s (~%d bytes)", img, size)


def _safe_workspace_path(base: str, name: str) -> str:
    """Resolve a workspace-relative file name, rejecting path traversal."""
    dest = os.path.normpath(os.path.join(base, name))
    if dest != base and not dest.startswith(base + os.sep):
        raise ValueError(f"unsafe workspace file path: {name!r}")
    return dest


def _stage_files(dest_dir: str, files: dict) -> None:
    """Write a ``{name: content}`` mapping into ``dest_dir``.

    ``content`` is UTF-8 text (str) or a ``{"b64": "..."}`` wrapper for binary
    payloads. Nested names create subdirectories; path traversal is rejected.
    """
    for name, content in (files or {}).items():
        dest = _safe_workspace_path(dest_dir, name)
        os.makedirs(os.path.dirname(dest) or dest_dir, exist_ok=True)
        if isinstance(content, dict) and "b64" in content:
            with open(dest, "wb") as fh:
                fh.write(base64.b64decode(content["b64"]))
        else:
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write(content if isinstance(content, str) else str(content))


def _truncate_output(text: str, limit: int) -> str:
    """Bound a captured output string, keeping head and tail around a marker."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - limit
    return text[:head] + f"\n... [{omitted} bytes truncated] ...\n" + text[-tail:]


def _build_key(dockerfile: str, context_files: dict, build_args: dict) -> str:
    """Content hash identifying a build, so identical environments dedupe."""
    h = hashlib.sha256()
    h.update(dockerfile.encode("utf-8"))
    for name in sorted(context_files or {}):
        c = context_files[name]
        h.update(b"\0")
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        if isinstance(c, dict) and "b64" in c:
            h.update(c["b64"].encode("utf-8"))
        else:
            h.update((c if isinstance(c, str) else str(c)).encode("utf-8"))
    for k in sorted(build_args or {}):
        h.update(f"\0{k}={build_args[k]}".encode("utf-8"))
    return h.hexdigest()[:24]


def _ensure_built(
    build_key: str,
    dockerfile: str,
    context_files: dict,
    build_args: dict,
    timeout: int,
) -> str:
    """Build (once) and reserve a local image tag from a Dockerfile + context.

    Reuses the same LRU bookkeeping as pulled images (``_image_in_use`` /
    ``_image_last_used``), so the caller releases with ``_release_image``.
    Returns the resolved image ref. If the tag already exists it is reused
    without rebuilding.
    """
    ref = f"dockyard-build:{build_key}"
    with _image_lock:
        _image_in_use[ref] = _image_in_use.get(ref, 0) + 1
        _image_last_used[ref] = time.monotonic()
    if _image_present(ref):
        return ref
    build_dir = os.path.join(TASK_BASE_DIR, f"build_{build_key}_{uuid.uuid4().hex[:8]}")
    try:
        os.makedirs(build_dir, exist_ok=True)
        dockerfile_path = os.path.join(build_dir, "Dockerfile.dockyard")
        with open(dockerfile_path, "w", encoding="utf-8") as fh:
            fh.write(dockerfile)
        _stage_files(build_dir, context_files)
        cmd = [DOCKER_BIN, "build", "-t", ref, "-f", dockerfile_path]
        for k, v in (build_args or {}).items():
            cmd += ["--build-arg", f"{k}={v}"]
        cmd += [build_dir]
        rc, _out, err = _run(cmd, timeout=timeout)
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)
    if rc != 0:
        with _image_lock:
            n = _image_in_use.get(ref, 1) - 1
            if n <= 0:
                _image_in_use.pop(ref, None)
            else:
                _image_in_use[ref] = n
        raise RuntimeError(f"docker build failed (rc={rc}): {err.strip()[:800]}")
    return ref


def _copy_out_file(container: str, path: str) -> str | None:
    """Copy a single file out of a container and return its text, or None."""
    tmp = os.path.join(TASK_BASE_DIR, f"cp_{uuid.uuid4().hex[:8]}")
    try:
        rc, _out, _err = _run([DOCKER_BIN, "cp", f"{container}:{path}", tmp], timeout=60)
        if rc != 0:
            return None
        with open(tmp, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# ===========================================================================
# Interactive session support (multi-turn terminal tasks)
# ===========================================================================
# A session is a long-lived container an agent drives across many turns:
#   POST /session/start          → 202 {session_id, status:"starting"}
#   GET  /session/<id>           → {session_id, status, image, error}
#   POST /session/<id>/exec      → run one command, return stdout/stderr/rc
#   POST /session/<id>/finish    → late-inject a verifier harness, run it,
#                                   read back result files, tear the session down
#   DELETE /session/<id>         → force teardown
#
# The image is acquired (pull or build) for the session's whole lifetime and
# released exactly once at teardown. Tests are never present during the agent
# turns — they are `docker cp`-ed in at finish time — so the agent cannot
# tamper with the held-out verifier. Orphaned sessions (crashed rollouts) are
# reaped by age.

_SESSION_MAX_AGE = int(os.environ.get("DOCKYARD_SESSION_MAX_AGE", "1800"))
_EXEC_DEFAULT_TIMEOUT = int(os.environ.get("DOCKYARD_SESSION_EXEC_TIMEOUT", "60"))
_SESSION_OUTPUT_MAX = int(os.environ.get("DOCKYARD_SESSION_OUTPUT_MAX_BYTES", "65536"))

_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _session_container(sid: str) -> str:
    return f"dockyard_sess_{sid}"


def _teardown_session(sid: str) -> bool:
    """Atomically remove a session: stop its container, release its image.

    The session record is popped under the lock, so exactly one caller (finish,
    DELETE, or the reaper) wins and performs teardown. Returns False if the
    session was already gone.
    """
    with _sessions_lock:
        s = _sessions.pop(sid, None)
        if s is None:
            return False
        image = s.get("image") if s.get("image_held") else None
        s["image_held"] = False
        container = s.get("container") or _session_container(sid)
    _run([DOCKER_BIN, "rm", "-f", container], timeout=30)
    if image is not None:
        _release_image(image)
    return True


def _session_bootstrap(sid: str, params: dict) -> None:
    """Acquire the image (pull or build) and start the keep-alive container."""
    mode          = params.get("mode", "image")
    block_network = bool(params.get("block_network", False))
    env_vars      = params.get("env") or {}
    cap_drop      = params.get("cap_drop") or []
    startup       = params.get("startup_command")
    pull_timeout  = int(params.get("pull_timeout", _IMAGE_PULL_TIMEOUT))
    build_timeout = int(params.get("build_timeout", _IMAGE_PULL_TIMEOUT))
    container     = _session_container(sid)

    ref: str | None = None
    try:
        if not _docker_available():
            raise RuntimeError("docker daemon is not available in the sandbox")

        if mode == "image":
            ref = str(params["image"])
            _ensure_image(ref, timeout=pull_timeout)
        elif mode == "build":
            key = _build_key(
                params["dockerfile"],
                params.get("build_context") or {},
                params.get("build_args") or {},
            )
            ref = _ensure_built(
                key,
                params["dockerfile"],
                params.get("build_context") or {},
                params.get("build_args") or {},
                build_timeout,
            )
        else:
            raise RuntimeError(f"unsupported session mode: {mode!r}")

        # Record the held image. If the session was deleted during acquisition,
        # release immediately and stop.
        proceed = False
        with _sessions_lock:
            if sid in _sessions:
                _sessions[sid]["image"] = ref
                _sessions[sid]["image_held"] = True
                proceed = True
        if not proceed:
            _release_image(ref)
            return

        run_cmd = [DOCKER_BIN, "run", "-d", "--init", "--name", container]
        if block_network:
            run_cmd += ["--network", "none"]
        for cap in cap_drop:
            run_cmd += ["--cap-drop", str(cap)]
        for k, v in env_vars.items():
            run_cmd += ["-e", f"{k}={v}"]
        run_cmd += ["--entrypoint", "/bin/bash", ref, "-c", "sleep infinity"]
        rc, _out, err = _run(run_cmd, timeout=120)
        if rc != 0:
            raise RuntimeError(f"docker run failed (rc={rc}): {err.strip()[:500]}")

        if startup:
            # Non-fatal background startup (e.g. a service the task needs up).
            _run(
                [DOCKER_BIN, "exec", "-d", container, "/bin/bash", "-lc", startup],
                timeout=30,
            )

        done = False
        with _sessions_lock:
            if sid in _sessions:
                _sessions[sid].update(
                    status="ready", container=container, _last_used=time.monotonic()
                )
                done = True
        if not done:
            # Deleted during startup: clean up the container and release. A
            # racing DELETE may also release; _release_image floors at zero so
            # the double-decrement is harmless.
            _run([DOCKER_BIN, "rm", "-f", container], timeout=30)
            _release_image(ref)

    except Exception as exc:
        log.exception("Session %s bootstrap failed: %s", sid, exc)
        rel: str | None = None
        with _sessions_lock:
            s = _sessions.get(sid)
            if s is not None:
                rel = s.get("image") if s.get("image_held") else None
                s["image_held"] = False
                s["status"] = "failed"
                s["error"] = str(exc)
        _run([DOCKER_BIN, "rm", "-f", container], timeout=30)
        if rel is not None:
            _release_image(rel)
        elif ref is not None and s is None:
            # Image was acquired but the record vanished before image_held was
            # set; release the local ref directly.
            _release_image(ref)


def _session_exec(sid: str, params: dict) -> dict:
    """Run a single command in a session's container."""
    with _sessions_lock:
        s = _sessions.get(sid)
        if s is None:
            return {"_http": HTTPStatus.NOT_FOUND, "error": "session not found"}
        if s["status"] != "ready":
            return {
                "_http": HTTPStatus.CONFLICT,
                "error": f"session not ready (status={s['status']})",
            }
        container = s["container"]
        image = s["image"]
        s["_last_used"] = time.monotonic()

    command   = params.get("command", "")
    timeout   = int(params.get("timeout", _EXEC_DEFAULT_TIMEOUT))
    cwd       = params.get("cwd")
    out_limit = int(params.get("output_limit", _SESSION_OUTPUT_MAX))

    cmd = [DOCKER_BIN, "exec"]
    if cwd:
        cmd += ["-w", cwd]
    cmd += [container, "/bin/bash", "-lc", command]

    timed_out = False
    try:
        rc, out, err = _run(cmd, timeout=timeout)
    except _TaskTimeoutError:
        timed_out = True
        rc, out, err = 124, "", f"command timed out after {timeout}s"

    with _image_lock:
        if image in _image_last_used:
            _image_last_used[image] = time.monotonic()

    return {
        "session_id": sid,
        "exit_code":  rc,
        "stdout":     _truncate_output(out, out_limit),
        "stderr":     _truncate_output(err, out_limit),
        "timed_out":  timed_out,
    }


def _session_finish(sid: str, params: dict) -> dict:
    """Inject a verifier harness, run it, read result files, tear down."""
    with _sessions_lock:
        s = _sessions.get(sid)
        if s is None:
            return {"_http": HTTPStatus.NOT_FOUND, "error": "session not found"}
        if s["status"] not in ("ready",):
            return {
                "_http": HTTPStatus.CONFLICT,
                "error": f"session not ready (status={s['status']})",
            }
        s["status"] = "finishing"
        s["_last_used"] = time.monotonic()
        container = s["container"]

    harness_files = params.get("harness_files") or {}
    harness_mount = params.get("harness_mount") or "/verifier"
    test_command  = params.get("test_command") or f"bash {harness_mount}/test.sh"
    result_files  = params.get("result_files") or []
    timeout       = int(params.get("timeout", 600))
    out_limit     = int(params.get("output_limit", _SESSION_OUTPUT_MAX))

    start     = time.monotonic()
    stage_dir = os.path.join(TASK_BASE_DIR, f"finish_{sid}_{uuid.uuid4().hex[:8]}")
    collected: dict[str, str | None] = {}
    rc, out, err = -1, "", ""
    timed_out = False
    error: str | None = None

    try:
        if harness_files:
            os.makedirs(stage_dir, exist_ok=True)
            _stage_files(stage_dir, harness_files)
            _run(
                [DOCKER_BIN, "exec", container, "/bin/bash", "-lc",
                 f"mkdir -p {harness_mount}"],
                timeout=30,
            )
            cp_rc, _o, cp_err = _run(
                [DOCKER_BIN, "cp", f"{stage_dir}/.", f"{container}:{harness_mount}"],
                timeout=120,
            )
            if cp_rc != 0:
                raise RuntimeError(f"harness injection failed: {cp_err.strip()[:300]}")

        try:
            rc, out, err = _run(
                [DOCKER_BIN, "exec", container, "/bin/bash", "-lc", test_command],
                timeout=timeout,
            )
        except _TaskTimeoutError:
            timed_out = True
            rc, out, err = 124, "", f"verifier timed out after {timeout}s"

        for path in result_files:
            collected[path] = _copy_out_file(container, path)

    except Exception as exc:
        error = str(exc)
        log.exception("Session %s finish failed: %s", sid, exc)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        _teardown_session(sid)

    return {
        "session_id":     sid,
        "exit_code":      rc,
        "stdout":         _truncate_output(out, out_limit),
        "stderr":         _truncate_output(err, out_limit),
        "result_files":   collected,
        "timed_out":      timed_out,
        "error":          error,
        "execution_time": round(time.monotonic() - start, 3),
    }


def _session_export(sid: str, params: dict) -> dict:
    """Export a session workspace path as a base64 gzip tar (does not tear down).

    Used to capture an agent's submission before grading. The tar is produced by
    the container's own tar, streamed to the host, and returned base64-encoded.
    """
    with _sessions_lock:
        s = _sessions.get(sid)
        if s is None:
            return {"_http": HTTPStatus.NOT_FOUND, "error": "session not found"}
        if s["status"] not in ("ready",):
            return {
                "_http": HTTPStatus.CONFLICT,
                "error": f"session not ready (status={s['status']})",
            }
        container = s["container"]
        s["_last_used"] = time.monotonic()

    path    = params.get("path") or "/workspace"
    timeout = int(params.get("timeout", 300))
    tmp     = os.path.join(TASK_BASE_DIR, f"export_{sid}_{uuid.uuid4().hex[:8]}.tar.gz")
    try:
        # Stream the container's own tar to a host file: `docker exec` writes the
        # archive to stdout, captured via shell redirection on the host side.
        with open(tmp, "wb") as fh:
            proc = subprocess.Popen(
                [DOCKER_BIN, "exec", container, "tar", "-C", path, "-czf", "-", "."],
                stdout=fh,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                _, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                return {"_http": HTTPStatus.INTERNAL_SERVER_ERROR,
                        "error": f"export timed out after {timeout}s"}
        if proc.returncode != 0:
            return {"_http": HTTPStatus.INTERNAL_SERVER_ERROR,
                    "error": f"tar export failed: {err.decode(errors='replace')[:300]}"}
        with open(tmp, "rb") as fh:
            data = fh.read()
        return {"session_id": sid, "path": path, "tar_b64": base64.b64encode(data).decode("ascii")}
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _reap_sessions_once() -> None:
    """Tear down sessions idle longer than _SESSION_MAX_AGE (orphan cleanup)."""
    cutoff = time.monotonic() - _SESSION_MAX_AGE
    with _sessions_lock:
        stale = [
            sid for sid, s in _sessions.items()
            if s.get("_last_used", s.get("_created", 0)) < cutoff
        ]
    for sid in stale:
        log.warning("Reaping stale session %s", sid)
        _teardown_session(sid)


def _session_reaper_loop() -> None:
    while True:
        time.sleep(60)
        try:
            _reap_sessions_once()
        except Exception as exc:
            log.warning("session reaper error: %s", exc)


# ===========================================================================
# Core task executor
# ===========================================================================

def _patch_target_paths(repo_dir: str, patch_file: str) -> list[str]:
    """Return the repo-relative file paths a unified diff touches.

    Parses ``git apply --numstat``, whose lines are tab-separated
    ``<added> <deleted> <path>``. Rename lines ("old => new") are normalised
    to the new path. Returns an empty list if the patch cannot be parsed.
    """
    rc, out, _ = _run(
        ["git", "apply", "--numstat", patch_file], cwd=repo_dir, timeout=30
    )
    if rc != 0:
        return []
    paths: list[str] = []
    for line in out.splitlines():
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        path = cols[2].strip()
        if " => " in path:
            # "{old => new}" or "dir/{a => b}/f" → keep the new-path side.
            path = path.replace("{", "").replace("}", "")
            path = path.split(" => ")[-1].replace("//", "/")
        if path:
            paths.append(path)
    return paths


def _restore_paths_to_base(repo_dir: str, base_commit: str, patch_file: str) -> None:
    """Reset the files a patch targets to their base_commit state.

    For each path the patch touches: if it exists at base_commit, restore that
    version (discarding working-tree edits); otherwise remove any working-tree
    copy so the patch can add it cleanly. This guarantees the patch applies onto
    a canonical base regardless of prior working-tree modifications to those
    paths.
    """
    for path in _patch_target_paths(repo_dir, patch_file):
        exists_rc, _, _ = _run(
            ["git", "cat-file", "-e", f"{base_commit}:{path}"],
            cwd=repo_dir,
            timeout=15,
        )
        if exists_rc == 0:
            _run(["git", "checkout", base_commit, "--", path], cwd=repo_dir, timeout=30)
        else:
            full = os.path.join(repo_dir, path)
            try:
                if os.path.exists(full):
                    os.remove(full)
            except OSError:
                pass


def _execute(task_id: str, params: dict) -> None:
    """Dispatch a submitted task to the clone or image executor by `mode`."""
    if params.get("mode", "clone") == "image":
        _execute_image(task_id, params)
        return
    _execute_clone(
        task_id,
        params["repo_url"],
        params.get("base_commit", "HEAD"),
        params.get("patch", ""),
        params.get("test_patch", ""),
        params["test_command"],
        int(params.get("timeout", 300)),
        params.get("lint_command", ""),
        bool(params.get("capture_diff", False)),
        params.get("reference_patch", ""),
    )


def _execute_image(task_id: str, params: dict) -> None:
    """Run a prebuilt per-instance image task with an isolated harness.

    Staged `harness_files` are mounted READ-ONLY at `harness_mount` (default
    /harness) so agent code running during the task cannot rewrite the runner
    or any gold harness file on disk. A separate writable dir is mounted at
    `output_mount` (default /out) for the run's outputs; `result_files` are read
    back from there. `entry_command` runs under `/bin/bash -c`. The image is
    pulled on demand and tracked for LRU eviction. Returns the raw output-file
    contents plus the container's stdout/stderr/exit code; interpretation
    (parsing, scoring) is the reward's responsibility and runs outside the
    container.
    """
    image          = params["image"]
    harness_files  = params.get("harness_files") or params.get("workspace_files") or {}
    harness_mount  = params.get("harness_mount") or "/harness"
    output_mount   = params.get("output_mount") or "/out"
    entry_command  = params.get("entry_command") or f"bash {harness_mount}/entryscript.sh"
    result_files   = params.get("result_files") or []
    timeout        = int(params.get("timeout", 300))
    block_network  = bool(params.get("block_network", False))
    pull_timeout   = int(params.get("pull_timeout", _IMAGE_PULL_TIMEOUT))

    task_dir    = os.path.join(TASK_BASE_DIR, task_id)
    harness_dir = os.path.join(task_dir, "harness")
    output_dir  = os.path.join(task_dir, "out")
    container   = f"dockyard_{task_id}"
    start       = time.monotonic()

    final_update: dict = {
        "status":         "failed",
        "exit_code":      -1,
        "stdout":         "",
        "stderr":         "",
        "result_files":   {},
        "image":          image,
        "execution_time": 0.0,
    }

    def _update(**kw: object) -> None:
        with _tasks_lock:
            _tasks[task_id].update(kw)

    _update(status="running")

    acquired = False
    try:
        if not _docker_available():
            raise RuntimeError("docker daemon is not available in the sandbox")

        os.makedirs(harness_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        # The container user (often root) must be able to write results into
        # the bind-mounted output dir owned by the executor user.
        os.chmod(output_dir, 0o777)
        for name, content in harness_files.items():
            dest = _safe_workspace_path(harness_dir, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write(content if isinstance(content, str) else str(content))

        _ensure_image(image, timeout=pull_timeout)
        acquired = True

        cmd = [
            DOCKER_BIN, "run", "--rm", "--name", container,
            "-v", f"{harness_dir}:{harness_mount}:ro",
            "-v", f"{output_dir}:{output_mount}",
            "--entrypoint", "/bin/bash",
        ]
        if block_network:
            cmd += ["--network", "none"]
        cmd += [image, "-c", entry_command]

        try:
            rc, out, err = _run(cmd, timeout=timeout)
        except _TaskTimeoutError:
            _run([DOCKER_BIN, "rm", "-f", container], timeout=30)
            raise

        collected: dict[str, str | None] = {}
        for name in result_files:
            try:
                with open(_safe_workspace_path(output_dir, name), encoding="utf-8") as fh:
                    collected[name] = fh.read()
            except OSError:
                collected[name] = None

        final_update = {
            "status":         "completed",
            "exit_code":      rc,
            "stdout":         out,
            "stderr":         err,
            "result_files":   collected,
            "image":          image,
            "execution_time": round(time.monotonic() - start, 3),
        }

    except _TaskTimeoutError as exc:
        log.error("Image task %s timed out after %ds", task_id, timeout)
        final_update.update({
            "stderr":         str(exc),
            "execution_time": round(time.monotonic() - start, 3),
        })

    except Exception as exc:
        log.exception("Image task %s failed: %s", task_id, exc)
        final_update.update({
            "stderr":         str(exc),
            "execution_time": round(time.monotonic() - start, 3),
        })

    finally:
        if acquired:
            _release_image(image)
        _update(**final_update)
        shutil.rmtree(task_dir, ignore_errors=True)


def _execute_clone(
    task_id:          str,
    repo_url:         str,
    base_commit:      str,
    patch:            str,
    test_patch:       str,
    test_command:     str,
    timeout:          int,
    lint_command:     str,
    capture_diff:     bool,
    reference_patch:  str,
) -> None:
    task_dir        = os.path.join(TASK_BASE_DIR, task_id)
    repo_dir        = os.path.join(task_dir, "repo")
    patch_file      = os.path.join(task_dir, "task.patch")
    test_patch_file = os.path.join(task_dir, "test.patch")

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    start = time.monotonic()

    final_update: dict = {
        "status":           "failed",
        "exit_code":        -1,
        "stdout":           "",
        "stderr":           "",
        "tests_passed":     0,
        "tests_failed":     0,
        "lint_errors":      None,
        "lint_output":      None,
        "patch_diff":       None,
        "patch_similarity": None,
        "execution_time":   0.0,
        "patch_applied":      False,
        "test_patch_applied": False,
    }

    def _update(**kw: object) -> None:
        with _tasks_lock:
            _tasks[task_id].update(kw)

    _update(status="running")

    try:
        os.makedirs(task_dir, exist_ok=True)

        rc, out, err = _run(["git", "clone", repo_url, repo_dir], timeout=120)
        stdout_parts.append(out); stderr_parts.append(err)
        if rc != 0:
            raise RuntimeError(f"git clone failed (rc={rc}): {err.strip()}")

        rc, out, err = _run(["git", "checkout", base_commit], cwd=repo_dir, timeout=60)
        stdout_parts.append(out); stderr_parts.append(err)
        if rc != 0:
            raise RuntimeError(f"git checkout failed (rc={rc}): {err.strip()}")

        patch_applied = True
        if patch and patch.strip():
            with open(patch_file, "w", encoding="utf-8") as fh:
                fh.write(patch)
            rc, out, err = _run(["git", "apply", patch_file], cwd=repo_dir, timeout=30)
            stdout_parts.append(out); stderr_parts.append(err)
            if rc != 0:
                # Reported via patch_applied=False rather than raised: keep the
                # tree at base_commit and proceed so test_command still runs.
                patch_applied = False
                log.info("Task %s agent patch did not apply (rc=%d)", task_id, rc)

        # Reset the test_patch's target paths to base_commit, then apply the
        # gold test_patch. The held-out test files are made canonical regardless
        # of any modifications the agent patch made to them; paths outside the
        # test_patch are untouched.
        test_patch_applied = True
        if test_patch and test_patch.strip():
            with open(test_patch_file, "w", encoding="utf-8") as fh:
                fh.write(test_patch)
            _restore_paths_to_base(repo_dir, base_commit, test_patch_file)
            rc, out, err = _run(
                ["git", "apply", test_patch_file], cwd=repo_dir, timeout=30
            )
            stdout_parts.append(out); stderr_parts.append(err)
            if rc != 0:
                # The gold test_patch failing to apply is a data/infra error
                # (it is expected to apply cleanly onto the reset paths).
                test_patch_applied = False
                raise RuntimeError(f"test_patch apply failed (rc={rc}): {err.strip()}")

        rc, out, err = _run(test_command, cwd=repo_dir, timeout=timeout, shell=True)
        stdout_parts.append(out); stderr_parts.append(err)
        test_exit_code = rc

        combined_stdout = "\n".join(filter(None, stdout_parts))
        combined_stderr = "\n".join(filter(None, stderr_parts))
        passed, failed  = _dispatch_test_parser(
            test_command, combined_stdout + "\n" + combined_stderr
        )

        lint_errors_count: int | None = None
        lint_out:          str | None = None

        if lint_command and lint_command.strip():
            try:
                lint_rc, l_out, l_err = _run(
                    lint_command, cwd=repo_dir, timeout=120, shell=True
                )
                lint_out          = (l_out + "\n" + l_err).strip() or None
                lint_errors_count = _parse_lint_errors(
                    lint_command, lint_out or "", lint_rc
                )
                log.info("Task %s lint finished — rc=%d errors=%s",
                         task_id, lint_rc, lint_errors_count)
            except _TaskTimeoutError:
                lint_out          = "Lint timed out after 120s"
                lint_errors_count = None
                log.warning("Task %s lint timed out", task_id)
            except Exception as exc:
                lint_out          = f"Lint error: {exc}"
                lint_errors_count = None
                log.warning("Task %s lint exception: %s", task_id, exc)

        patch_diff_text: str | None = None

        if capture_diff or (reference_patch and reference_patch.strip()):
            try:
                _, diff_out, _ = _run(
                    ["git", "diff", base_commit], cwd=repo_dir, timeout=30
                )
                patch_diff_text = diff_out.strip() or None
            except Exception as exc:
                log.warning("Task %s git diff failed: %s", task_id, exc)

        similarity: float | None = None

        if reference_patch and reference_patch.strip():
            try:
                agent_diff = patch_diff_text or (patch if patch and patch.strip() else "")
                if agent_diff:
                    similarity = round(_patch_similarity(agent_diff, reference_patch), 4)
                    log.info("Task %s patch_similarity=%.4f", task_id, similarity)
            except Exception as exc:
                log.warning("Task %s similarity computation failed: %s", task_id, exc)

        final_update = {
            "status":           "completed",
            "exit_code":        test_exit_code,
            "stdout":           combined_stdout,
            "stderr":           combined_stderr,
            "tests_passed":     passed,
            "tests_failed":     failed,
            "lint_errors":      lint_errors_count,
            "lint_output":      lint_out,
            "patch_diff":       patch_diff_text,
            "patch_similarity": similarity,
            "execution_time":   round(time.monotonic() - start, 3),
            "patch_applied":      patch_applied,
            "test_patch_applied": test_patch_applied,
        }

    except _TaskTimeoutError as exc:
        stderr_parts.append(str(exc))
        log.error("Task %s timed out after %ds", task_id, timeout)
        final_update.update({
            "stdout":         "\n".join(filter(None, stdout_parts)),
            "stderr":         "\n".join(filter(None, stderr_parts)),
            "execution_time": round(time.monotonic() - start, 3),
        })

    except Exception as exc:
        stderr_parts.append(str(exc))
        log.exception("Task %s failed: %s", task_id, exc)
        final_update.update({
            "stdout":         "\n".join(filter(None, stdout_parts)),
            "stderr":         "\n".join(filter(None, stderr_parts)),
            "execution_time": round(time.monotonic() - start, 3),
        })

    finally:
        _update(**final_update)
        try:
            shutil.rmtree(task_dir, ignore_errors=True)
        except Exception:
            pass


# ===========================================================================
# REST endpoints
# ===========================================================================

@app.route("/task/submit", methods=["POST"])
def submit():
    """
    POST /task/submit

    Body (JSON):
        repo_url          str   required
        base_commit       str   optional  (default: HEAD)
        patch             str   optional
        test_patch        str   optional  (gold test diff, applied last)
        test_command      str   required
        timeout           int   optional  (default: 300)
        lint_command      str   optional
        capture_diff      bool  optional  (default: false)
        reference_patch   str   optional

    Returns 202: { "task_id": "<uuid>", "status": "pending" }

    Image mode (mode="image"): pull a prebuilt image and run a container with a
    read-only harness mount and a separate writable output mount.
        mode              "image"
        image             str   required  (full image ref to pull/run)
        harness_files     dict  optional  ({name: content}, mounted read-only)
        harness_mount     str   optional  (default: /harness)
        output_mount      str   optional  (default: /out, writable)
        entry_command     str   optional  (default: bash <harness_mount>/entryscript.sh)
        result_files      list  optional  (read back from the output mount)
        timeout           int   optional  (default: 300)
        block_network     bool  optional  (default: false → network enabled)
        pull_timeout      int   optional  (image pull timeout, seconds)
    """
    if not _check_auth():
        return jsonify(error="Unauthorized"), HTTPStatus.UNAUTHORIZED

    _evict_old_tasks()

    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify(error="Request body must be valid JSON"), HTTPStatus.BAD_REQUEST

    mode = body.get("mode", "clone")
    if mode not in ("clone", "image"):
        return jsonify(error=f"Unsupported mode: {mode!r}"), HTTPStatus.BAD_REQUEST

    required = ("image",) if mode == "image" else ("repo_url", "test_command")
    missing = [f for f in required if not body.get(f)]
    if missing:
        return jsonify(error=f"Missing required fields: {missing}"), HTTPStatus.BAD_REQUEST

    task_id = str(uuid.uuid4())
    record: dict = {
        "task_id":          task_id,
        "status":           "pending",
        "_created":         time.monotonic(),
        "mode":             mode,
        "repo_url":         body.get("repo_url"),
        "base_commit":      body.get("base_commit", "HEAD"),
        "test_command":     body.get("test_command"),
        "image":            body.get("image"),
        "timeout":          int(body.get("timeout", 300)),
        "exit_code":        None,
        "stdout":           None,
        "stderr":           None,
        "tests_passed":     None,
        "tests_failed":     None,
        "lint_errors":      None,
        "lint_output":      None,
        "patch_diff":       None,
        "patch_similarity": None,
        "execution_time":   None,
        "patch_applied":      None,
        "test_patch_applied": None,
        "result_files":     None,
    }

    with _tasks_lock:
        _tasks[task_id] = record

    threading.Thread(
        target=_execute,
        args=(task_id, body),
        daemon=True,
    ).start()

    log.info(
        "Task %s submitted — mode=%s target=%s",
        task_id, mode, body.get("image") if mode == "image" else body.get("repo_url"),
    )
    return jsonify(task_id=task_id, status="pending"), HTTPStatus.ACCEPTED


@app.route("/task/<task_id>", methods=["GET"])
def status(task_id: str):
    if not _check_auth():
        return jsonify(error="Unauthorized"), HTTPStatus.UNAUTHORIZED
    with _tasks_lock:
        t = _tasks.get(task_id)
    if t is None:
        return jsonify(error="Task not found"), HTTPStatus.NOT_FOUND
    return jsonify(task_id=t["task_id"], status=t["status"])


@app.route("/task/<task_id>/result", methods=["GET"])
def result(task_id: str):
    if not _check_auth():
        return jsonify(error="Unauthorized"), HTTPStatus.UNAUTHORIZED
    with _tasks_lock:
        t = _tasks.get(task_id)
    if t is None:
        return jsonify(error="Task not found"), HTTPStatus.NOT_FOUND
    if t["status"] in ("pending", "running"):
        return jsonify(
            task_id=task_id,
            status=t["status"],
            message="Task not yet complete — poll again shortly",
        ), HTTPStatus.ACCEPTED
    return jsonify({k: v for k, v in t.items() if not str(k).startswith("_")})


@app.route("/task/<task_id>", methods=["DELETE"])
def delete(task_id: str):
    if not _check_auth():
        return jsonify(error="Unauthorized"), HTTPStatus.UNAUTHORIZED
    with _tasks_lock:
        if task_id not in _tasks:
            return jsonify(error="Task not found"), HTTPStatus.NOT_FOUND
        _tasks.pop(task_id)
    log.info("Task %s deleted", task_id)
    return jsonify(task_id=task_id, deleted=True)


# ---------------------------------------------------------------------------
# Interactive session endpoints (multi-turn terminal tasks)
# ---------------------------------------------------------------------------

@app.route("/session/start", methods=["POST"])
def session_start():
    """
    POST /session/start

    Body (JSON):
        mode            "image" (default) | "build"
        block_network   bool  optional  (default: false)
        cap_drop        list  optional  (Linux capabilities to drop, e.g. ["SYS_PTRACE"])
        env             dict  optional  (container environment variables)
        startup_command str   optional  (best-effort background start command)
        pull_timeout    int   optional  (image-mode pull timeout, seconds)
        build_timeout   int   optional  (build-mode build timeout, seconds)
      image mode:
        image           str   required  (full image ref to pull/run)
      build mode:
        dockerfile      str   required  (Dockerfile content)
        build_context   dict  optional  ({name: str|{"b64":...}} build context)
        build_args      dict  optional  (Docker --build-arg key/values)

    Returns 202: {session_id, status:"starting"}. Poll GET /session/<id> until
    status is "ready" (or "failed").
    """
    if not _check_auth():
        return jsonify(error="Unauthorized"), HTTPStatus.UNAUTHORIZED

    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify(error="Request body must be valid JSON"), HTTPStatus.BAD_REQUEST

    mode = body.get("mode", "image")
    if mode not in ("image", "build"):
        return jsonify(error=f"Unsupported mode: {mode!r}"), HTTPStatus.BAD_REQUEST
    required = ("image",) if mode == "image" else ("dockerfile",)
    missing = [f for f in required if not body.get(f)]
    if missing:
        return jsonify(error=f"Missing required fields: {missing}"), HTTPStatus.BAD_REQUEST

    sid = str(uuid.uuid4())
    now = time.monotonic()
    with _sessions_lock:
        _sessions[sid] = {
            "session_id": sid,
            "status":     "starting",
            "mode":       mode,
            "image":      None,
            "image_held": False,
            "container":  None,
            "error":      None,
            "_created":   now,
            "_last_used": now,
        }

    threading.Thread(target=_session_bootstrap, args=(sid, body), daemon=True).start()
    log.info("Session %s starting — mode=%s", sid, mode)
    return jsonify(session_id=sid, status="starting"), HTTPStatus.ACCEPTED


@app.route("/session/<sid>", methods=["GET"])
def session_status(sid: str):
    if not _check_auth():
        return jsonify(error="Unauthorized"), HTTPStatus.UNAUTHORIZED
    with _sessions_lock:
        s = _sessions.get(sid)
        if s is None:
            return jsonify(error="Session not found"), HTTPStatus.NOT_FOUND
        return jsonify(
            session_id=sid, status=s["status"], image=s.get("image"),
            error=s.get("error"),
        )


@app.route("/session/<sid>/exec", methods=["POST"])
def session_exec_ep(sid: str):
    if not _check_auth():
        return jsonify(error="Unauthorized"), HTTPStatus.UNAUTHORIZED
    body = request.get_json(force=True, silent=True) or {}
    result = _session_exec(sid, body)
    http = result.pop("_http", HTTPStatus.OK)
    return jsonify(result), http


@app.route("/session/<sid>/export", methods=["POST"])
def session_export_ep(sid: str):
    if not _check_auth():
        return jsonify(error="Unauthorized"), HTTPStatus.UNAUTHORIZED
    body = request.get_json(force=True, silent=True) or {}
    result = _session_export(sid, body)
    http = result.pop("_http", HTTPStatus.OK)
    return jsonify(result), http


@app.route("/session/<sid>/finish", methods=["POST"])
def session_finish_ep(sid: str):
    if not _check_auth():
        return jsonify(error="Unauthorized"), HTTPStatus.UNAUTHORIZED
    body = request.get_json(force=True, silent=True) or {}
    result = _session_finish(sid, body)
    http = result.pop("_http", HTTPStatus.OK)
    return jsonify(result), http


@app.route("/session/<sid>", methods=["DELETE"])
def session_delete_ep(sid: str):
    if not _check_auth():
        return jsonify(error="Unauthorized"), HTTPStatus.UNAUTHORIZED
    with _sessions_lock:
        s = _sessions.get(sid)
        if s is None:
            return jsonify(error="Session not found"), HTTPStatus.NOT_FOUND
        if s["status"] == "starting":
            # Bootstrap owns image acquisition; refuse until it settles.
            return jsonify(
                error="cannot delete during startup; retry after status is ready/failed"
            ), HTTPStatus.CONFLICT
    ok = _teardown_session(sid)
    log.info("Session %s deleted", sid)
    return jsonify(session_id=sid, deleted=ok)


if __name__ == "__main__":
    log.info("Task executor starting on 0.0.0.0:%d", API_PORT)
    threading.Thread(target=_session_reaper_loop, daemon=True).start()
    serve(app, host="0.0.0.0", port=API_PORT, threads=_API_THREADS)
