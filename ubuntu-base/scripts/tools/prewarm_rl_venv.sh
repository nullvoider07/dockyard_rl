#!/bin/bash
#
# Pre-warms the Ray worker venv on each node before Ray starts.
# Prevents the uv venv stampede that occurs when RAY_ENABLE_UV_RUN_RUNTIME_ENV=1
# causes all workers to race to build the same venv on first task.
#
# Run this on every node before `ray start`, either via:
#   - A Kubernetes initContainer
#   - A docker-compose entrypoint hook
#   - Manually before launching the Ray head/worker
#
# Idempotent: skips creation if a valid venv already exists at RL_VENV_DIR.
# Safe for concurrent execution: uses a lockfile to serialize across processes
# that might race on the same node (e.g. multiple containers sharing a volume).

set -euo pipefail

# Config
VENV_DIR="${RL_VENV_DIR:-/opt/rl_venvs}"
VENV_NAME="${RL_VENV_NAME:-base}"
VENV_PATH="${VENV_DIR}/${VENV_NAME}"
PYTHON_BIN="${RL_PYTHON_BIN:-python3}"
LOCK_FILE="${VENV_DIR}/.prewarm.lock"
LOG_PREFIX="[prewarm_rl_venv]"

# Packages to pre-install in the venv.
# Override by mounting a requirements file at PREWARM_REQUIREMENTS_FILE,
# or by setting PREWARM_EXTRA_PACKAGES as a space-separated list.
PREWARM_REQUIREMENTS_FILE="${PREWARM_REQUIREMENTS_FILE:-}"
PREWARM_EXTRA_PACKAGES="${PREWARM_EXTRA_PACKAGES:-}"

# Logging
log()  { echo "${LOG_PREFIX} $*"; }
info() { echo "${LOG_PREFIX} [INFO]  $*"; }
warn() { echo "${LOG_PREFIX} [WARN]  $*" >&2; }
err()  { echo "${LOG_PREFIX} [ERROR] $*" >&2; }

# Preflight
if ! command -v uv &>/dev/null; then
    err "uv not found. Cannot pre-warm venv."
    err "Ensure uv is installed at /usr/local/bin/uv (see Dockerfile §4)."
    exit 1
fi

mkdir -p "${VENV_DIR}"

# Lock — serialize across racing processes on the same node
# flock -n: fail immediately if lock is held (another process is pre-warming).
# flock -w 120: wait up to 120s (another process is already pre-warming; wait for it).
(
    flock -w 120 200 || {
        err "Timed out waiting for prewarm lock after 120s. Another process may have stalled."
        exit 1
    }

    # Check: skip if venv already valid
    if [ -f "${VENV_PATH}/bin/python" ] && \
       [ -f "${VENV_PATH}/.prewarm-complete" ]; then
        BAKED_HASH=$(cat "${VENV_PATH}/.prewarm-complete" 2>/dev/null || echo "")
        CURRENT_HASH=$(${PYTHON_BIN} --version 2>&1 | sha256sum | cut -c1-8)
        if [ "${BAKED_HASH}" = "${CURRENT_HASH}" ]; then
            info "Valid venv found at ${VENV_PATH}. Skipping creation."
            exit 0
        else
            warn "Python version changed. Rebuilding venv at ${VENV_PATH}."
            rm -rf "${VENV_PATH}"
        fi
    elif [ -d "${VENV_PATH}" ]; then
        warn "Incomplete venv found at ${VENV_PATH}. Rebuilding."
        rm -rf "${VENV_PATH}"
    fi

    # Create venv
    info "Creating venv at ${VENV_PATH} using Python: $(${PYTHON_BIN} --version)"
    uv venv --python "${PYTHON_BIN}" "${VENV_PATH}"

    # Install packages
    # Always install the core Ray stack first — these are the deps Ray workers
    # universally need regardless of task type.
    info "Installing core Ray dependencies..."
    uv pip install --python "${VENV_PATH}/bin/python" \
        "ray[default]" \
        psutil \
        pyyaml

    # Install from requirements file if provided
    if [ -n "${PREWARM_REQUIREMENTS_FILE}" ]; then
        if [ -f "${PREWARM_REQUIREMENTS_FILE}" ]; then
            info "Installing from requirements file: ${PREWARM_REQUIREMENTS_FILE}"
            uv pip install --python "${VENV_PATH}/bin/python" \
                -r "${PREWARM_REQUIREMENTS_FILE}"
        else
            warn "PREWARM_REQUIREMENTS_FILE set but file not found: ${PREWARM_REQUIREMENTS_FILE}"
        fi
    fi

    # Install any extra packages passed via env var
    if [ -n "${PREWARM_EXTRA_PACKAGES}" ]; then
        info "Installing extra packages: ${PREWARM_EXTRA_PACKAGES}"
        # Word-split intentional here — PREWARM_EXTRA_PACKAGES is space-separated
        # shellcheck disable=SC2086
        uv pip install --python "${VENV_PATH}/bin/python" ${PREWARM_EXTRA_PACKAGES}
    fi

    # Stamp completion
    # Write a hash of the Python version so stale venvs are caught on Python upgrades.
    ${PYTHON_BIN} --version 2>&1 | sha256sum | cut -c1-8 > "${VENV_PATH}/.prewarm-complete"
    chown -R 1001:1001 "${VENV_PATH}" 2>/dev/null || true

    info "Venv pre-warm complete: ${VENV_PATH}"
    uv pip list --python "${VENV_PATH}/bin/python" | head -20
    info "(truncated — full list at: uv pip list --python ${VENV_PATH}/bin/python)"

) 200>"${LOCK_FILE}"