"""dockyard_rl.sandbox — client for the ubuntu-swe task executor.

The task executor (Component 1, ubuntu-base/scripts/tools/task_executor.py)
exposes a batch task-submission API. This package is the trainer-side client
both the environment and the reward layer use to drive it; the agent never
reaches the executor directly.
"""

from dockyard_rl.sandbox.task_executor_client import (
    ExecResult,
    FinishResult,
    SessionStartSpec,
    TaskExecutorError,
    TaskResult,
    TaskSpec,
    delete_session,
    delete_task,
    get_result,
    get_session_status,
    run_task,
    session_exec,
    session_export,
    session_finish,
    start_session,
    submit_task,
)

__all__ = [
    "ExecResult",
    "FinishResult",
    "SessionStartSpec",
    "TaskExecutorError",
    "TaskResult",
    "TaskSpec",
    "delete_session",
    "delete_task",
    "get_result",
    "get_session_status",
    "run_task",
    "session_exec",
    "session_export",
    "session_finish",
    "start_session",
    "submit_task",
]
