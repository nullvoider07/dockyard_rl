"""RDMA rail discovery for the Mooncake transport, kept import-light.

Split out of ``adapters/transfer_queue.py`` deliberately. That module imports
``transfer_queue`` and ``tensordict`` at module level, neither of which is
installed in this tree or listed in ``ubuntu-base/uv.lock``, so anything living
inside it can only be exercised on a machine that has both — which is why a
piece of pure sysfs-reading logic went untested. Nothing here needs either
package.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


INFINIBAND_SYSFS = "/sys/class/infiniband"


def discover_roce_devices(sysfs_root: str = INFINIBAND_SYSFS) -> list[str]:
    """RoCE-capable RDMA device names under ``sysfs_root``, sorted.

    Replaces a ``for d in .../mlx5_*/ports/1/link_layer; ... | head -1`` shell
    pipeline with the same selection — ``mlx5`` devices whose ``link_layer``
    reads ``Ethernet`` — expressed so it can be exercised against a fake sysfs
    tree instead of only on a node that has the hardware.

    Two differences from the pipeline, both free: every port is read rather than
    port 1 alone, so a dual-port card whose second port is the RoCE one is no
    longer invisible; and the result is sorted rather than left to glob order.

    RoCE-only is deliberate, not an oversight — the caller pins
    ``MC_GID_INDEX=3``, the RoCEv2 GID index, so an InfiniBand rail would need
    different GID handling and is excluded rather than half-supported.
    """
    try:
        entries = sorted(os.listdir(sysfs_root))
    except OSError:
        return []

    devices: list[str] = []
    for name in entries:
        if not name.startswith("mlx5_"):
            continue
        ports_dir = os.path.join(sysfs_root, name, "ports")
        try:
            ports = sorted(os.listdir(ports_dir))
        except OSError:
            continue
        for port in ports:
            link_layer = os.path.join(ports_dir, port, "link_layer")
            try:
                with open(link_layer) as handle:
                    if "Ethernet" in handle.read():
                        devices.append(name)
                        break
            except OSError:
                continue
    return devices


def _mooncake_transport_config() -> dict:
    protocol = os.environ.get("MC_MOONCAKE_PROTOCOL", "tcp")
    if protocol != "rdma":
        return {"protocol": "tcp"}

    device = os.environ.get("MC_MOONCAKE_DEVICE", "")
    if not device:
        discovered = discover_roce_devices()
        if not discovered:
            # Previously this fell through silently with device_name="", which
            # reads as a configured RDMA transport and is not one.
            logger.warning(
                "MC_MOONCAKE_PROTOCOL=rdma but no RoCE device was found under "
                "%s. Mooncake will be handed an empty device_name and left to "
                "its own rail selection. Set MC_MOONCAKE_DEVICE if this node "
                "has an RDMA rail.",
                INFINIBAND_SYSFS,
            )
            return {"protocol": "rdma", "device_name": ""}
        device = discovered[0]
        if len(discovered) > 1:
            # Offering every rail (upstream #2935, guarded there by
            # MC_ENABLE_DEST_DEVICE_AFFINITY) needs to know whether mooncake's
            # device_name accepts a list. Neither mooncake nor transfer_queue
            # is installed in this tree or its image lock, so that is not
            # checkable here and the single-rail choice is reported instead of
            # silently made.
            logger.info(
                "RoCE rails found: %s. Using %s; the others are not offered.",
                ", ".join(discovered),
                device,
            )

    os.environ.setdefault("MC_GID_INDEX", "3")
    return {"protocol": "rdma", "device_name": device}
