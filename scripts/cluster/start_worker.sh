#!/bin/bash
# Project Dockyard — Ray worker node bootstrap.
#
# Use this script when you are NOT using Slurm (i.e. bare-metal, Docker,
# or direct SSH access to nodes).  For Slurm deployments, use ray.sub at
# the repo root instead.
#
# This script attaches a worker node to an existing Ray head and declares
# the node's fleet role via DOCKYARD_FLEET_ROLE.  The role is read at
# runtime by dockyard_rl.cluster.bootstrap.get_fleet_role(), which
# determines which placement group strategy and resource spec to use.
#
# Usage:
#   DOCKYARD_FLEET_ROLE=<trainer|inference|sandbox> \
#   RAY_ADDRESS=<head-ip>:<port> \
#   bash scripts/cluster/start_worker.sh [OPTIONS]
#
# Options (all optional — defaults shown):
#   --fleet-role ROLE      trainer | inference | sandbox
#                          Overrides DOCKYARD_FLEET_ROLE env var.
#   --head-address ADDR    head-ip:port (overrides RAY_ADDRESS env var)
#   --gpus-per-node N      GPUs on this node (default: auto-detected)
#   --log-dir PATH         Directory for Ray logs (default: /tmp/dockyard/logs)
#   --block                Block until Ray exits (default: true)
#
# Environment variables:
#   DOCKYARD_FLEET_ROLE    Required. trainer | inference | sandbox.
#   RAY_ADDRESS            Required. Address of the Ray head node.
#   GPUS_PER_NODE          GPUs per node (overrides auto-detection).
#   DOCKYARD_LOG_DIR       Log directory.
#   NCCL_SOCKET_IFNAME     Network interface for NCCL (e.g. eth0, ib0).
#
# Example — trainer worker joining a head at 10.0.0.1:6379:
#   DOCKYARD_FLEET_ROLE=trainer \
#   RAY_ADDRESS=10.0.0.1:6379 \
#   bash scripts/cluster/start_worker.sh --gpus-per-node 8
#
# Example — sandbox (CPU-only) worker:
#   DOCKYARD_FLEET_ROLE=sandbox \
#   RAY_ADDRESS=10.0.0.1:6379 \
#   bash scripts/cluster/start_worker.sh --gpus-per-node 0

set -euo pipefail

# Argument parsing
FLEET_ROLE="${DOCKYARD_FLEET_ROLE:-}"
HEAD_ADDRESS="${RAY_ADDRESS:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-}"
DOCKYARD_LOG_DIR="${DOCKYARD_LOG_DIR:-/tmp/dockyard/logs}"
BLOCK=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fleet-role)      FLEET_ROLE="$2";        shift 2 ;;
        --head-address)    HEAD_ADDRESS="$2";       shift 2 ;;
        --gpus-per-node)   GPUS_PER_NODE="$2";     shift 2 ;;
        --log-dir)         DOCKYARD_LOG_DIR="$2";   shift 2 ;;
        --no-block)        BLOCK=false;             shift   ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            echo "Usage: DOCKYARD_FLEET_ROLE=<role> RAY_ADDRESS=<addr> bash scripts/cluster/start_worker.sh [OPTIONS]" >&2
            exit 1
            ;;
    esac
done

# Validate required inputs
if [[ -z "$FLEET_ROLE" ]]; then
    echo "[ERROR] DOCKYARD_FLEET_ROLE is not set." >&2
    echo "  Set it via env var or --fleet-role <trainer|inference|sandbox>" >&2
    exit 1
fi

case "$FLEET_ROLE" in
    trainer|inference|sandbox) ;;
    *)
        echo "[ERROR] Invalid DOCKYARD_FLEET_ROLE=$FLEET_ROLE." >&2
        echo "  Valid values: trainer, inference, sandbox" >&2
        exit 1
        ;;
esac

if [[ -z "$HEAD_ADDRESS" ]]; then
    # Try reading from file written by start_head.sh.
    ADDR_FILE="${DOCKYARD_LOG_DIR}/ray_head_address"
    if [[ -f "$ADDR_FILE" ]]; then
        HEAD_ADDRESS=$(cat "$ADDR_FILE")
        echo "[INFO] Read head address from $ADDR_FILE: $HEAD_ADDRESS"
    else
        echo "[ERROR] RAY_ADDRESS is not set and $ADDR_FILE does not exist." >&2
        echo "  Set RAY_ADDRESS=<head-ip>:<port> or run start_head.sh first." >&2
        exit 1
    fi
fi

mkdir -p "$DOCKYARD_LOG_DIR"

# GPU auto-detection
if [[ -z "$GPUS_PER_NODE" ]]; then
    if [[ "$FLEET_ROLE" == "sandbox" ]]; then
        # Sandbox fleet is CPU-only; never claim GPU resources.
        GPUS_PER_NODE=0
    elif command -v nvidia-smi &>/dev/null; then
        GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    else
        GPUS_PER_NODE=0
    fi
    echo "[INFO] Auto-detected GPUs per node: $GPUS_PER_NODE"
fi

# Sandbox fleet must not claim GPUs regardless of what was passed.
if [[ "$FLEET_ROLE" == "sandbox" ]] && [[ "$GPUS_PER_NODE" -gt 0 ]]; then
    echo "[WARN] sandbox fleet must be CPU-only. Overriding --gpus-per-node to 0."
    GPUS_PER_NODE=0
fi

# Environment
# Fleet role forwarded into every Ray worker actor spawned on this node.
export DOCKYARD_FLEET_ROLE="$FLEET_ROLE"
export RAY_ADDRESS="$HEAD_ADDRESS"

# All deps are in ubuntu-swe system Python — no uv venv stampede.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# Prevents false OOM kills from Ray's memory monitor.
export RAY_memory_monitor_refresh_ms=0

# GPU ordering: match NVML physical order so RANK → GPU mapping is stable.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Prevent deadlock with Ray fork-based workers.
export TOKENIZERS_PARALLELISM=false

# Prevent CPU thread oversubscription inside Ray workers.
export OMP_NUM_THREADS=1

# NCCL interface detection — same logic as start_head.sh.
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
    PRIMARY_IF=$(ip -o -4 route show to default 2>/dev/null \
        | awk '{print $5}' | head -1 || true)
    if [[ -n "$PRIMARY_IF" ]]; then
        export NCCL_SOCKET_IFNAME="$PRIMARY_IF"
        echo "[INFO] Auto-set NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    fi
fi

# Ulimit
if [[ $(ulimit -Hn) == "unlimited" ]] || [[ 65535 -lt $(ulimit -Hn) ]]; then
    ulimit -Sn 65535 2>/dev/null || true
fi

# Port configuration
# Worker ports must not collide with the head (which uses +1 offsets).
NODE_MANAGER_PORT=${NODE_MANAGER_PORT:-53001}
OBJECT_MANAGER_PORT=${OBJECT_MANAGER_PORT:-53003}
RUNTIME_ENV_AGENT_PORT=${RUNTIME_ENV_AGENT_PORT:-53005}
DASHBOARD_AGENT_GRPC_PORT=${DASHBOARD_AGENT_GRPC_PORT:-53007}
DASHBOARD_AGENT_LISTEN_PORT=${DASHBOARD_AGENT_LISTEN_PORT:-52365}
METRICS_EXPORT_PORT=${METRICS_EXPORT_PORT:-53009}
MIN_WORKER_PORT=${MIN_WORKER_PORT:-54001}
MAX_WORKER_PORT=${MAX_WORKER_PORT:-54513}

# Log file name
WORKER_LOG="$DOCKYARD_LOG_DIR/ray-worker-${FLEET_ROLE}-$$.log"

# Start Ray worker
echo ""
echo "============================================================"
echo " Starting Ray WORKER node"
echo "  Fleet role : $FLEET_ROLE"
echo "  Head addr  : $HEAD_ADDRESS"
echo "  GPUs       : $GPUS_PER_NODE"
echo "============================================================"

RAY_START_ARGS=(
    --address="$HEAD_ADDRESS"
    --disable-usage-stats
    --resources="{\"worker_units\": ${GPUS_PER_NODE}, \"dockyard_managed_ray_cluster\": 1, \"dockyard_fleet_${FLEET_ROLE}\": 1}"
    --node-manager-port="$NODE_MANAGER_PORT"
    --object-manager-port="$OBJECT_MANAGER_PORT"
    --runtime-env-agent-port="$RUNTIME_ENV_AGENT_PORT"
    --dashboard-agent-grpc-port="$DASHBOARD_AGENT_GRPC_PORT"
    --dashboard-agent-listen-port="$DASHBOARD_AGENT_LISTEN_PORT"
    --metrics-export-port="$METRICS_EXPORT_PORT"
    --min-worker-port="$MIN_WORKER_PORT"
    --max-worker-port="$MAX_WORKER_PORT"
)

if [[ "$BLOCK" == "true" ]]; then
    RAY_START_ARGS+=(--block)
fi

ray start "${RAY_START_ARGS[@]}" \
    2>&1 | tee "$WORKER_LOG"

echo ""
echo "============================================================"
echo " Ray worker node ($FLEET_ROLE) started."
echo " Log: $WORKER_LOG"
echo "============================================================"