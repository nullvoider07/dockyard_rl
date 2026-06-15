#!/bin/bash
set -euo pipefail

TASK_BASE_DIR="${TASK_BASE_DIR:-/workspace/tasks}"
API_PORT="${API_PORT:-9090}"
TASK_MAX_AGE="${TASK_MAX_AGE:-3600}"

mkdir -p "$TASK_BASE_DIR"

export TASK_BASE_DIR API_PORT TASK_MAX_AGE

# Best-effort wait for dockerd so image-mode (SWE-bench Pro) tasks do not race a
# still-initialising daemon. Non-fatal: clone-mode tasks need no docker, and the
# executor rechecks docker availability per task.
DOCKER_WAIT="${DOCKYARD_DOCKER_WAIT_SECONDS:-30}"
if command -v docker >/dev/null 2>&1; then
    for _ in $(seq 1 "$DOCKER_WAIT"); do
        if docker info >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
fi

exec python3 /usr/local/lib/task-executor/task_executor.py