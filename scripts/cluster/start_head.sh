#!/bin/bash
# Project Dockyard — Ray head node bootstrap.
#
# Use this script when you are NOT using Slurm (i.e. bare-metal, Docker,
# or direct SSH access to nodes).  For Slurm deployments, use ray.sub at
# the repo root instead.
#
# This script starts the Ray head node on the current machine and prints
# the RAY_ADDRESS that worker nodes need to join the cluster.
#
# Usage:
#   bash scripts/cluster/start_head.sh [OPTIONS]
#
# Options (all optional — defaults shown):
#   --gpus-per-node N      GPUs on this node (default: auto-detected)
#   --port N               Ray head GCS port (default: 6379)
#   --dashboard-port N     Ray dashboard port (default: 8265)
#   --log-dir PATH         Directory for Ray logs (default: /tmp/dockyard/logs)
#   --block                Block until Ray exits (default: false)
#
# Environment variables (override options):
#   GPUS_PER_NODE          GPUs per node
#   RAY_PORT               Ray GCS port
#   DASHBOARD_PORT         Ray dashboard port
#   DOCKYARD_LOG_DIR       Log directory
#   NCCL_SOCKET_IFNAME     Network interface for NCCL (e.g. eth0, ib0)
#                          Set this when using InfiniBand or RoCE.
#
# Example — 4-node cluster, head on node0:
#   # On node0 (head):
#   bash scripts/cluster/start_head.sh --gpus-per-node 8
#
#   # On node1, node2, node3 (workers):
#   RAY_ADDRESS=<node0-ip>:6379 \
#   DOCKYARD_FLEET_ROLE=trainer \
#   bash scripts/cluster/start_worker.sh --gpus-per-node 8

set -euo pipefail

# Argument parsing
GPUS_PER_NODE="${GPUS_PER_NODE:-}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
DOCKYARD_LOG_DIR="${DOCKYARD_LOG_DIR:-/tmp/dockyard/logs}"
BLOCK=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus-per-node)   GPUS_PER_NODE="$2";   shift 2 ;;
        --port)            RAY_PORT="$2";          shift 2 ;;
        --dashboard-port)  DASHBOARD_PORT="$2";    shift 2 ;;
        --log-dir)         DOCKYARD_LOG_DIR="$2";  shift 2 ;;
        --block)           BLOCK=true;             shift   ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            echo "Usage: bash scripts/cluster/start_head.sh [--gpus-per-node N] [--port N] [--dashboard-port N] [--log-dir PATH] [--block]" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$DOCKYARD_LOG_DIR"

# GPU auto-detection
if [[ -z "$GPUS_PER_NODE" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    else
        GPUS_PER_NODE=0
    fi
    echo "[INFO] Auto-detected GPUs per node: $GPUS_PER_NODE"
fi

# Environment variables and Ray configuration
# These must be set before ray start so they are visible to all Ray workers
# spawned on this node.

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

# NCCL: if NCCL_SOCKET_IFNAME is already set (e.g. ib0 for InfiniBand),
# keep it.  Otherwise, detect the primary non-loopback interface.
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
    PRIMARY_IF=$(ip -o -4 route show to default 2>/dev/null \
        | awk '{print $5}' | head -1 || true)
    if [[ -n "$PRIMARY_IF" ]]; then
        export NCCL_SOCKET_IFNAME="$PRIMARY_IF"
        echo "[INFO] Auto-set NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    fi
fi

# NCCL debugging aids — enable only if NCCL_DEBUG is already set externally.
# Uncomment to force: export NCCL_DEBUG=INFO
# Uncomment for InfiniBand: export NCCL_IB_DISABLE=0

# Node IP discovery
HEAD_IP=$(python3 -c "import ray._private.services as s; print(s.get_node_ip_address())" 2>/dev/null \
    || hostname -I | awk '{print $1}')

echo "[INFO] Head node IP: $HEAD_IP"
echo "[INFO] Ray GCS port: $RAY_PORT"
echo "[INFO] Dashboard port: $DASHBOARD_PORT"
echo "[INFO] GPUs per node: $GPUS_PER_NODE"

# Ulimit
# Ray best practices: raise the open-file-descriptor limit.
if [[ $(ulimit -Hn) == "unlimited" ]] || [[ 65535 -lt $(ulimit -Hn) ]]; then
    ulimit -Sn 65535 2>/dev/null || true
fi

# Start Ray head
RAY_START_ARGS=(
    --head
    --disable-usage-stats
    --node-ip-address="$HEAD_IP"
    --port="$RAY_PORT"
    --dashboard-port="$DASHBOARD_PORT"
    --dashboard-host="$HEAD_IP"
    --include-dashboard=True
    --resources="{\"worker_units\": ${GPUS_PER_NODE}, \"dockyard_managed_ray_cluster\": 1}"
)

if [[ "$BLOCK" == "true" ]]; then
    RAY_START_ARGS+=(--block)
fi

echo ""
echo "============================================================"
echo " Starting Ray HEAD node"
echo "  RAY_ADDRESS=${HEAD_IP}:${RAY_PORT}"
echo "============================================================"

ray start "${RAY_START_ARGS[@]}" \
    2>&1 | tee "$DOCKYARD_LOG_DIR/ray-head.log"

# Export address for workers
# Write a small file that worker nodes can read to get the head address
# without requiring manual copy-paste.
ADDR_FILE="$DOCKYARD_LOG_DIR/ray_head_address"
echo "${HEAD_IP}:${RAY_PORT}" > "$ADDR_FILE"

echo ""
echo "============================================================"
echo " Ray head node started."
echo ""
echo " To join worker nodes, run on each worker:"
echo "   RAY_ADDRESS=${HEAD_IP}:${RAY_PORT} \\"
echo "   DOCKYARD_FLEET_ROLE=<trainer|inference|sandbox> \\"
echo "   bash scripts/cluster/start_worker.sh --gpus-per-node $GPUS_PER_NODE"
echo ""
echo " Head address also written to: $ADDR_FILE"
echo " Dashboard: http://${HEAD_IP}:${DASHBOARD_PORT}"
echo "============================================================"