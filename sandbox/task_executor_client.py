"""Trainer-side HTTP client for the ubuntu-swe task executor.

The executor (ubuntu-base/scripts/tools/task_executor.py) is a batch
task-submission service:

    POST   /task/submit          → 202 {task_id, status}
    GET    /task/<id>            → {task_id, status}
    GET    /task/<id>/result     → 202 while pending/running,
                                   200 with the full record when finished
    DELETE /task/<id>            → evict the record

A task describes a self-contained evaluation: the executor clones
``repo_url`` fresh, checks out ``base_commit``, applies ``patch`` via
``git apply``, runs ``test_command`` (shell), parses the framework output,
optionally lints and computes patch similarity against ``reference_patch``,
then deletes the clone. Only this trainer-side client drives the executor —
the agent never reaches it.

Authentication: when the executor is started with ``API_TOKEN`` set, every
request must carry ``Authorization: Bearer <token>``. The shipped deployment
leaves auth disabled; when enabled, the token defaults to the
``DOCKYARD_SANDBOX_API_TOKEN`` environment variable.
"""

from __future__ import annotations

import os
import time
from typing import NotRequired, Optional, TypedDict, cast

import requests

# Shared connection pool; callers may inject their own Session instead.
_DEFAULT_SESSION = requests.Session()

# Environment variable carrying the executor bearer token (optional).
_API_TOKEN_ENV = "DOCKYARD_SANDBOX_API_TOKEN"


class TaskSpec(TypedDict, total=False):
    """Request body for ``POST /task/submit``.

    Clone mode (the default, ``mode`` absent or "clone"): ``repo_url`` and
    ``test_command`` are required; the rest match the executor's documented
    defaults (base_commit="HEAD", patch="", timeout=300, lint_command="",
    capture_diff=False, reference_patch="").

    Image mode (``mode="image"``): ``image`` is required. The executor pulls the
    image on demand, stages ``harness_files`` on a READ-ONLY mount (default
    /harness) and a separate writable output mount (default /out), runs
    ``entry_command`` under ``/bin/bash -c``, and reads ``result_files`` back
    from the output mount. Benchmark-specific assembly and interpretation live
    in the reward, not the executor.
    """

    # clone mode
    repo_url:        str
    test_command:    str
    base_commit:     NotRequired[str]
    patch:           NotRequired[str]
    test_patch:      NotRequired[str]
    timeout:         NotRequired[int]
    lint_command:    NotRequired[str]
    capture_diff:    NotRequired[bool]
    reference_patch: NotRequired[str]
    # image mode
    mode:            NotRequired[str]
    image:           NotRequired[str]
    # Values are UTF-8 text (str) or a {"b64": "..."} wrapper for binary payloads.
    harness_files:   NotRequired[dict[str, object]]
    harness_mount:   NotRequired[str]
    output_mount:    NotRequired[str]
    entry_command:   NotRequired[str]
    result_files:    NotRequired[list[str]]
    block_network:   NotRequired[bool]
    pull_timeout:    NotRequired[int]


class TaskResult(TypedDict):
    """Result record returned by ``GET /task/<id>/result`` once finished.

    ``status`` is "completed" when the task ran to completion (even if tests
    failed) or "failed" on an infrastructure error/timeout. ``exit_code`` is
    the ``test_command`` exit status.
    """

    task_id:            str
    status:             str
    exit_code:          int
    stdout:             str
    stderr:             str
    tests_passed:       Optional[int]
    tests_failed:       Optional[int]
    lint_errors:        Optional[int]
    lint_output:        Optional[str]
    patch_diff:         Optional[str]
    patch_similarity:   Optional[float]
    execution_time:     Optional[float]
    patch_applied:      Optional[bool]
    test_patch_applied: Optional[bool]
    # image mode
    mode:               NotRequired[str]
    image:              NotRequired[Optional[str]]
    result_files:       NotRequired[Optional[dict[str, Optional[str]]]]


class SessionStartSpec(TypedDict, total=False):
    """Request body for ``POST /session/start`` (multi-turn terminal tasks).

    Image mode (``mode`` absent or "image"): ``image`` is required and pulled on
    demand. Build mode (``mode="build"``): ``dockerfile`` is required and built
    once (cached by content hash); ``build_context`` carries sibling files
    ({name: str | {"b64": ...}}). The container is started detached with a
    keep-alive entrypoint; the agent drives it via ``/session/<id>/exec``.
    """

    mode:            NotRequired[str]
    image:           NotRequired[str]
    dockerfile:      NotRequired[str]
    build_context:   NotRequired[dict[str, object]]
    build_args:      NotRequired[dict[str, str]]
    env:             NotRequired[dict[str, str]]
    block_network:   NotRequired[bool]
    cap_drop:        NotRequired[list[str]]
    startup_command: NotRequired[str]
    pull_timeout:    NotRequired[int]
    build_timeout:   NotRequired[int]


class ExecResult(TypedDict):
    """Result of ``POST /session/<id>/exec``."""

    session_id: str
    exit_code:  int
    stdout:     str
    stderr:     str
    timed_out:  bool


class FinishResult(TypedDict):
    """Result of ``POST /session/<id>/finish``.

    ``result_files`` maps each requested container path to its text content (or
    None if it was absent). The session is torn down server-side after finish.
    """

    session_id:     str
    exit_code:      int
    stdout:         str
    stderr:         str
    result_files:   dict[str, Optional[str]]
    timed_out:      bool
    error:          Optional[str]
    execution_time: Optional[float]


class TaskExecutorError(RuntimeError):
    """Raised when the task executor is unreachable or returns an error."""


def _resolve_token(api_token: Optional[str]) -> str:
    return api_token if api_token is not None else os.environ.get(_API_TOKEN_ENV, "")


def _auth_headers(api_token: Optional[str]) -> dict[str, str]:
    token = _resolve_token(api_token)
    return {"Authorization": f"Bearer {token}"} if token else {}


def submit_task(
    base_url: str,
    spec: TaskSpec,
    *,
    api_token: Optional[str] = None,
    session: Optional[requests.Session] = None,
    request_timeout: float = 30.0,
) -> str:
    """Submit a task spec and return its ``task_id``.

    Raises:
        TaskExecutorError: on transport failure, non-202 status, or a
            response missing ``task_id``.
    """
    sess = session or _DEFAULT_SESSION
    try:
        resp = sess.post(
            f"{base_url}/task/submit",
            json=dict(spec),
            headers=_auth_headers(api_token),
            timeout=request_timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise TaskExecutorError(f"submit failed for {base_url}: {exc}") from exc

    if resp.status_code != 202:
        raise TaskExecutorError(
            f"submit returned HTTP {resp.status_code} from {base_url}: "
            f"{resp.text[:512]}"
        )

    task_id = resp.json().get("task_id")
    if not task_id:
        raise TaskExecutorError(
            f"submit response missing task_id from {base_url}: {resp.text[:512]}"
        )
    return cast(str, task_id)


def get_result(
    base_url: str,
    task_id: str,
    *,
    api_token: Optional[str] = None,
    session: Optional[requests.Session] = None,
    request_timeout: float = 30.0,
) -> Optional[TaskResult]:
    """Fetch a task's result, or ``None`` if it is still pending/running.

    Raises:
        TaskExecutorError: on transport failure or an unexpected HTTP status.
    """
    sess = session or _DEFAULT_SESSION
    try:
        resp = sess.get(
            f"{base_url}/task/{task_id}/result",
            headers=_auth_headers(api_token),
            timeout=request_timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise TaskExecutorError(
            f"result fetch failed for task {task_id} at {base_url}: {exc}"
        ) from exc

    if resp.status_code == 202:
        return None  # pending/running — poll again
    if resp.status_code != 200:
        raise TaskExecutorError(
            f"result returned HTTP {resp.status_code} for task {task_id}: "
            f"{resp.text[:512]}"
        )
    return cast(TaskResult, resp.json())


def delete_task(
    base_url: str,
    task_id: str,
    *,
    api_token: Optional[str] = None,
    session: Optional[requests.Session] = None,
    request_timeout: float = 30.0,
) -> None:
    """Best-effort eviction of a finished task's record. Never raises."""
    sess = session or _DEFAULT_SESSION
    try:
        sess.delete(
            f"{base_url}/task/{task_id}",
            headers=_auth_headers(api_token),
            timeout=request_timeout,
        )
    except requests.exceptions.RequestException:
        pass


def run_task(
    base_url: str,
    spec: TaskSpec,
    *,
    api_token: Optional[str] = None,
    session: Optional[requests.Session] = None,
    poll_interval: float = 2.0,
    overall_timeout: Optional[float] = None,
    cleanup: bool = True,
) -> TaskResult:
    """Submit a task and block until it finishes; return its result.

    ``overall_timeout`` defaults to the spec's ``timeout`` plus a 120s buffer
    to cover clone/checkout/apply/lint overhead beyond the test run itself. For
    image-mode specs the buffer also covers a cold image pull (``pull_timeout``,
    default 1800s), since a first-use pull blocks before the container runs.

    Raises:
        TaskExecutorError: on transport failure or if the task does not finish
            within ``overall_timeout``.
    """
    sess = session or _DEFAULT_SESSION
    if overall_timeout is None:
        buffer = 120.0
        if spec.get("mode") == "image":
            buffer += float(spec.get("pull_timeout", 1800))
        overall_timeout = float(spec.get("timeout", 300)) + buffer

    task_id = submit_task(base_url, spec, api_token=api_token, session=sess)
    deadline = time.monotonic() + overall_timeout
    try:
        while True:
            result = get_result(base_url, task_id, api_token=api_token, session=sess)
            if result is not None:
                return result
            if time.monotonic() >= deadline:
                raise TaskExecutorError(
                    f"task {task_id} did not finish within {overall_timeout:.0f}s"
                )
            time.sleep(poll_interval)
    finally:
        if cleanup:
            delete_task(base_url, task_id, api_token=api_token, session=sess)


# ---------------------------------------------------------------------------
# Interactive session client (multi-turn terminal tasks)
# ---------------------------------------------------------------------------

def get_session_status(
    base_url: str,
    session_id: str,
    *,
    api_token: Optional[str] = None,
    session: Optional[requests.Session] = None,
    request_timeout: float = 30.0,
) -> dict:
    """Return a session's status record ({session_id, status, image, error})."""
    sess = session or _DEFAULT_SESSION
    try:
        resp = sess.get(
            f"{base_url}/session/{session_id}",
            headers=_auth_headers(api_token),
            timeout=request_timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise TaskExecutorError(
            f"session status fetch failed for {session_id} at {base_url}: {exc}"
        ) from exc
    if resp.status_code != 200:
        raise TaskExecutorError(
            f"session status returned HTTP {resp.status_code} for {session_id}: "
            f"{resp.text[:512]}"
        )
    return cast(dict, resp.json())


def delete_session(
    base_url: str,
    session_id: str,
    *,
    api_token: Optional[str] = None,
    session: Optional[requests.Session] = None,
    request_timeout: float = 30.0,
) -> None:
    """Best-effort teardown of a session. Never raises."""
    sess = session or _DEFAULT_SESSION
    try:
        sess.delete(
            f"{base_url}/session/{session_id}",
            headers=_auth_headers(api_token),
            timeout=request_timeout,
        )
    except requests.exceptions.RequestException:
        pass


def start_session(
    base_url: str,
    spec: SessionStartSpec,
    *,
    api_token: Optional[str] = None,
    session: Optional[requests.Session] = None,
    poll_interval: float = 2.0,
    overall_timeout: Optional[float] = None,
    request_timeout: float = 30.0,
) -> str:
    """Start a session and block until it is ready; return its ``session_id``.

    ``overall_timeout`` defaults to the pull (image mode) or build (build mode)
    timeout plus a 120s buffer, since a cold pull/build blocks before the
    container is up.

    Raises:
        TaskExecutorError: on transport failure, a non-202 start, a failed
            bootstrap, or if the session is not ready within ``overall_timeout``.
    """
    sess = session or _DEFAULT_SESSION
    if overall_timeout is None:
        if spec.get("mode") == "build":
            overall_timeout = float(spec.get("build_timeout", 1800)) + 120.0
        else:
            overall_timeout = float(spec.get("pull_timeout", 1800)) + 120.0

    try:
        resp = sess.post(
            f"{base_url}/session/start",
            json=dict(spec),
            headers=_auth_headers(api_token),
            timeout=request_timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise TaskExecutorError(f"session start failed for {base_url}: {exc}") from exc
    if resp.status_code != 202:
        raise TaskExecutorError(
            f"session start returned HTTP {resp.status_code} from {base_url}: "
            f"{resp.text[:512]}"
        )
    session_id = resp.json().get("session_id")
    if not session_id:
        raise TaskExecutorError(
            f"session start response missing session_id from {base_url}: {resp.text[:512]}"
        )

    deadline = time.monotonic() + overall_timeout
    while True:
        status_rec = get_session_status(
            base_url, session_id, api_token=api_token, session=sess
        )
        status = status_rec.get("status")
        if status == "ready":
            return cast(str, session_id)
        if status == "failed":
            delete_session(base_url, session_id, api_token=api_token, session=sess)
            raise TaskExecutorError(
                f"session {session_id} failed to start: {status_rec.get('error')}"
            )
        if time.monotonic() >= deadline:
            delete_session(base_url, session_id, api_token=api_token, session=sess)
            raise TaskExecutorError(
                f"session {session_id} not ready within {overall_timeout:.0f}s"
            )
        time.sleep(poll_interval)


def session_exec(
    base_url: str,
    session_id: str,
    command: str,
    *,
    timeout: int = 60,
    cwd: Optional[str] = None,
    output_limit: Optional[int] = None,
    api_token: Optional[str] = None,
    session: Optional[requests.Session] = None,
    request_timeout: Optional[float] = None,
) -> ExecResult:
    """Run one command in a session and return its result.

    ``request_timeout`` defaults to ``timeout + 30s`` so the HTTP read outlasts
    the server-side command execution.
    """
    sess = session or _DEFAULT_SESSION
    if request_timeout is None:
        request_timeout = float(timeout) + 30.0
    body: dict[str, object] = {"command": command, "timeout": int(timeout)}
    if cwd:
        body["cwd"] = cwd
    if output_limit is not None:
        body["output_limit"] = int(output_limit)
    try:
        resp = sess.post(
            f"{base_url}/session/{session_id}/exec",
            json=body,
            headers=_auth_headers(api_token),
            timeout=request_timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise TaskExecutorError(
            f"session exec failed for {session_id} at {base_url}: {exc}"
        ) from exc
    if resp.status_code != 200:
        raise TaskExecutorError(
            f"session exec returned HTTP {resp.status_code} for {session_id}: "
            f"{resp.text[:512]}"
        )
    return cast(ExecResult, resp.json())


def session_export(
    base_url: str,
    session_id: str,
    *,
    path: str = "/workspace",
    timeout: int = 300,
    api_token: Optional[str] = None,
    session: Optional[requests.Session] = None,
    request_timeout: Optional[float] = None,
) -> str:
    """Export a session workspace path; return a base64 gzip-tar of its contents.

    Does not tear the session down (used to capture a submission before grading).
    """
    sess = session or _DEFAULT_SESSION
    if request_timeout is None:
        request_timeout = float(timeout) + 30.0
    body = {"path": path, "timeout": int(timeout)}
    try:
        resp = sess.post(
            f"{base_url}/session/{session_id}/export",
            json=body,
            headers=_auth_headers(api_token),
            timeout=request_timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise TaskExecutorError(
            f"session export failed for {session_id} at {base_url}: {exc}"
        ) from exc
    if resp.status_code != 200:
        raise TaskExecutorError(
            f"session export returned HTTP {resp.status_code} for {session_id}: "
            f"{resp.text[:512]}"
        )
    tar_b64 = resp.json().get("tar_b64")
    if not tar_b64:
        raise TaskExecutorError(
            f"session export response missing tar_b64 for {session_id}"
        )
    return cast(str, tar_b64)


def session_finish(
    base_url: str,
    session_id: str,
    *,
    harness_files: Optional[dict[str, object]] = None,
    harness_mount: Optional[str] = None,
    test_command: Optional[str] = None,
    result_files: Optional[list[str]] = None,
    timeout: int = 600,
    output_limit: Optional[int] = None,
    api_token: Optional[str] = None,
    session: Optional[requests.Session] = None,
    request_timeout: Optional[float] = None,
) -> FinishResult:
    """Inject a verifier harness, run it, read result files, tear down.

    ``request_timeout`` defaults to ``timeout + 60s`` to outlast the server-side
    verifier run plus harness staging.
    """
    sess = session or _DEFAULT_SESSION
    if request_timeout is None:
        request_timeout = float(timeout) + 60.0
    body: dict[str, object] = {"timeout": int(timeout)}
    if harness_files is not None:
        body["harness_files"] = harness_files
    if harness_mount:
        body["harness_mount"] = harness_mount
    if test_command:
        body["test_command"] = test_command
    if result_files is not None:
        body["result_files"] = result_files
    if output_limit is not None:
        body["output_limit"] = int(output_limit)
    try:
        resp = sess.post(
            f"{base_url}/session/{session_id}/finish",
            json=body,
            headers=_auth_headers(api_token),
            timeout=request_timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise TaskExecutorError(
            f"session finish failed for {session_id} at {base_url}: {exc}"
        ) from exc
    if resp.status_code != 200:
        raise TaskExecutorError(
            f"session finish returned HTTP {resp.status_code} for {session_id}: "
            f"{resp.text[:512]}"
        )
    return cast(FinishResult, resp.json())
