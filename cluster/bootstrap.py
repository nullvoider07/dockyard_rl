"""Ray cluster bootstrap.

Every container in the cluster declares its fleet role via
DOCKYARD_FLEET_ROLE before calling init_ray().  The rest of the
cluster-wiring code reads that role to decide which placement-group
strategy, resource spec, and NCCL topology to apply.

Two operating modes:
  Data-centre  — RAY_ADDRESS is set; attach unconditionally.
  Local-dev    — RAY_ADDRESS absent; reuse or start a local cluster
                 keyed by CUDA_VISIBLE_DEVICES for fast iteration.
"""

import importlib
import logging
import os
from typing import Optional

import ray

logger = logging.getLogger(__name__)

# Fleet role constants
FLEET_TRAINER   = "trainer"
FLEET_INFERENCE = "inference"
FLEET_SANDBOX   = "sandbox"

_ROLE_ENV    = "DOCKYARD_FLEET_ROLE"
_KNOWN_ROLES = frozenset({FLEET_TRAINER, FLEET_INFERENCE, FLEET_SANDBOX})

# Local-dev cluster tagging prefix.  Clusters started by this code carry
# a custom resource "dockyard_tag_<cvd>" so a second process with the same
# CUDA_VISIBLE_DEVICES can reuse the same local cluster without a cold start.
_TAG_PREFIX = "dockyard_tag_"

# Public API
def get_fleet_role() -> str:
    """Return the fleet role declared by DOCKYARD_FLEET_ROLE.

    Raises RuntimeError if the variable is absent, ValueError if the
    value is not one of the three recognised roles.
    """
    role = os.environ.get(_ROLE_ENV, "").strip().lower()
    if not role:
        raise RuntimeError(
            f"Environment variable {_ROLE_ENV!r} is not set. "
            f"Every container must declare its fleet role before calling "
            f"init_ray(). Valid values: {sorted(_KNOWN_ROLES)}"
        )
    if role not in _KNOWN_ROLES:
        raise ValueError(
            f"Unknown fleet role {role!r}. "
            f"Valid values: {sorted(_KNOWN_ROLES)}"
        )
    return role

def init_ray(log_dir: Optional[str] = None) -> None:
    """Connect this process to the Ray cluster.

    The full process environment is forwarded to every Ray worker so that
    image-level settings (DOCKYARD_FLEET_ROLE, NCCL_SOCKET_IFNAME,
    CUDA_DEVICE_ORDER, TOKENIZERS_PARALLELISM, …) are visible inside
    remote actors without any additional configuration.

    Args:
        log_dir: Optional path for Ray logs and temporary files.
                 Defaults to Ray's own default when None.
    """
    env_vars = dict(os.environ)
    # Ray sets this internally on workers; carrying it across causes
    # double-init errors on any worker that calls ray.init() itself.
    env_vars.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)
    runtime_env = {"env_vars": env_vars}

    temp_dir = os.path.abspath(log_dir) if log_dir else None

    ray_address = os.environ.get("RAY_ADDRESS", "").strip()
    if ray_address:
        _attach_datacenter(ray_address, runtime_env, temp_dir)
    else:
        _attach_or_start_local(runtime_env, temp_dir)

# Internal helpers
def _attach_datacenter(
    address: str,
    runtime_env: dict,
    temp_dir: Optional[str],
) -> None:
    """Attach to an externally managed cluster (Kubernetes, Slurm, etc.)."""
    logger.info("Attaching to Ray cluster at %s", address)
    ray.init(
        address=address,
        log_to_driver=True,
        include_dashboard=False,
        runtime_env=runtime_env,
        _temp_dir=temp_dir,
    )
    logger.info("Connected. Cluster resources: %s", ray.cluster_resources())

def _attach_or_start_local(
    runtime_env: dict,
    temp_dir: Optional[str],
) -> None:
    """Attempt to reuse a local cluster; start a fresh one if none exists.

    Clusters started by this function carry a custom resource
    "dockyard_tag_<cvd>" so a second process on the same machine whose
    CUDA_VISIBLE_DEVICES matches can reattach rather than cold-start.
    """
    cvd     = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
    cvd     = ",".join(sorted(cvd.split(",")))
    cvd_tag = f"{_TAG_PREFIX}{cvd.replace(',', '_')}"

    try:
        ray.init(
            address="auto",
            log_to_driver=True,
            include_dashboard=False,
            runtime_env=runtime_env,
            _temp_dir=temp_dir,
        )
        cluster_res = ray.cluster_resources()

        if any(k.startswith(_TAG_PREFIX) for k in cluster_res):
            # This is a Dockyard-managed local cluster.
            if cvd_tag in cluster_res:
                logger.info(
                    "Reusing local Ray cluster (CVD tag %r matched): %s",
                    cvd_tag, cluster_res,
                )
                return
            # CVD mismatch — a different Dockyard process owns it.
            logger.info(
                "Local cluster exists but CVD tag mismatch "
                "(expected %r, found %r). Starting a fresh cluster.",
                cvd_tag,
                next(k for k in cluster_res if k.startswith(_TAG_PREFIX)),
            )
            ray.shutdown()
            # Invalidate the driver-side package cache so working_dir is
            # re-uploaded if a future init() supplies one.
            importlib.reload(
                importlib.import_module("ray._private.runtime_env.packaging")
            )
        else:
            # Externally started local cluster (e.g. `ray start --head`).
            logger.info(
                "Connected to existing local Ray cluster: %s", cluster_res
            )
            return

    except ConnectionError:
        logger.debug("No existing local Ray cluster found; starting one.")
        ray.shutdown()

    _start_local_cluster(cvd_tag, runtime_env, temp_dir)

def _detect_node_resources() -> dict:
    """Auto-advertise custom Ray scheduling resources for this node.

    A ``kvm`` resource is declared when ``/dev/kvm`` is present, so KVM-bound
    actors — the OSWorld official backend runs a KVM-in-docker DesktopEnv on its
    own node — can pin to capable nodes via ``env.osworld.node_resources``. The
    slot count (DOCKYARD_KVM_SLOTS, default 8) bounds concurrent KVM episodes per
    node. Multi-node fleets declare the same on each worker's ``ray start``.
    """
    resources: dict = {}
    if os.path.exists("/dev/kvm"):
        try:
            resources["kvm"] = float(os.environ.get("DOCKYARD_KVM_SLOTS", "8").strip())
        except ValueError:
            resources["kvm"] = 8.0
    return resources


def _start_local_cluster(
    cvd_tag: str,
    runtime_env: dict,
    temp_dir: Optional[str],
) -> None:
    """Start a brand-new local single-node Ray cluster."""
    # Drop working_dir to avoid packaging the entire repo tree, which
    # triggers ray OSError: Failed to download runtime_env file package.
    local_env = {k: v for k, v in runtime_env.items() if k != "working_dir"}

    ray.init(
        log_to_driver=True,
        include_dashboard=True,
        runtime_env=local_env,
        _temp_dir=temp_dir,
        resources={cvd_tag: 1, **_detect_node_resources()},
    )
    logger.info(
        "Started local Ray cluster (tag %r): %s",
        cvd_tag, ray.cluster_resources(),
    )