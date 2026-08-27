"""Vendor-neutral device queries: UUID, memory, and visible-device resolution.

Two things live here, and they are together because the second is a
precondition for the first.

**The queries.** GPU UUID and free/total memory come from a vendor management
library — NVML on NVIDIA, and its counterpart on other vendors. This module
selects one at first use and exposes the queries without naming a vendor, so
importing ``dockyard_rl.utils`` no longer requires that a particular vendor's
Python bindings be installed. That import used to be unconditional and at module
scope, which meant a host without ``pynvml`` could not import the package at
all, well before any code had a chance to dispatch on what hardware it had.

**Visible-device resolution.** Management libraries index physical devices;
torch indexes the visible subset. Translating between them is a per-vendor rule,
not a variable rename, and reading it wrongly is worse than it sounds: the UUID
this module returns is what the colocated trainer and inference worker each use
to derive the refit IPC socket path (``models/policy/utils.refit_socket_address``),
so a wrong device answer does not raise — the two sides compute different paths
and the refit hangs.

The rule is torch's, read from ``torch.cuda._parse_visible_devices`` rather than
recalled: a HIP build starts from ``CUDA_VISIBLE_DEVICES``, lets
``HIP_VISIBLE_DEVICES`` override it, and applies ``ROCR_VISIBLE_DEVICES`` first
as a pool reduction — with ``HIP_VISIBLE_DEVICES`` preferred when both are set,
and refused when it names more devices than ROCR left. A CUDA build reads
``CUDA_VISIBLE_DEVICES`` alone.
"""

from __future__ import annotations

import contextlib
import os
from typing import Generator, MutableMapping, Optional, Protocol, runtime_checkable

import torch

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
HIP_VISIBLE_DEVICES = "HIP_VISIBLE_DEVICES"
ROCR_VISIBLE_DEVICES = "ROCR_VISIBLE_DEVICES"

# Every variable that can restrict which devices a process sees, in the order a
# reader should consider them. Consumers that propagate or clear the visible-set
# (worker launch, cache-key construction) must cover all of them or they cover
# NVIDIA only.
VISIBLE_DEVICE_ENV_VARS = (
    CUDA_VISIBLE_DEVICES,
    HIP_VISIBLE_DEVICES,
    ROCR_VISIBLE_DEVICES,
)

# Ray's per-vendor opt-outs from having it set the visible-device variable
# itself. Both names are read from the installed Ray
# (``_private/accelerators/{nvidia,amd}_gpu.py``); Ray sets HIP_VISIBLE_DEVICES
# on an AMD node, so a worker that suppresses only the CUDA one still has its
# devices reassigned underneath it.
NOSET_VISIBLE_DEVICE_ENV_VARS = (
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
)


@runtime_checkable
class DeviceQueryBackend(Protocol):
    """A vendor management library, reduced to what dockyard asks of it."""

    name: str

    def device_uuid(self, physical_index: int) -> str:
        """Stable identifier for a physical device."""
        ...

    def memory_info(self, physical_index: int) -> tuple[int, int]:
        """``(free, total)`` bytes for a physical device."""
        ...


class NvmlBackend:
    """NVIDIA Management Library, via ``pynvml`` (the ``nvidia-ml-py`` package).

    NVML is initialised and shut down around each query rather than held open:
    these are startup- and refit-time calls, not a hot path, and a handle held
    across a fork does not survive it.
    """

    name = "nvml"

    def __init__(self) -> None:
        import pynvml  # noqa: F401  -- fail here, not at first query

        self._pynvml = pynvml

    @contextlib.contextmanager
    def _session(self) -> Generator[None, None, None]:
        pynvml = self._pynvml
        try:
            pynvml.nvmlInit()
            yield
        except pynvml.NVMLError as exc:
            raise RuntimeError(f"Failed to initialise NVML: {exc}") from exc
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def device_uuid(self, physical_index: int) -> str:
        pynvml = self._pynvml
        with self._session():
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
            except pynvml.NVMLError as exc:
                raise RuntimeError(
                    f"Failed to get UUID for physical device {physical_index}: {exc}"
                ) from exc
        if isinstance(uuid, bytes):
            return uuid.decode("utf-8")
        if isinstance(uuid, str):
            return uuid
        raise RuntimeError(
            f"Unexpected UUID type {type(uuid)} for physical device {physical_index}."
        )

    def memory_info(self, physical_index: int) -> tuple[int, int]:
        pynvml = self._pynvml
        with self._session():
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            except pynvml.NVMLError as exc:
                raise RuntimeError(
                    f"Failed to get memory info for physical device "
                    f"{physical_index}: {exc}"
                ) from exc
        return int(info.free), int(info.total)


_BACKEND: Optional[DeviceQueryBackend] = None


def set_device_query_backend(backend: Optional[DeviceQueryBackend]) -> None:
    """Install a backend explicitly, or clear the cached one with ``None``."""
    global _BACKEND
    _BACKEND = backend


def resolve_device_query_backend() -> DeviceQueryBackend:
    """The management-library backend for this host, selected once.

    Selection is by torch's own build, not by what happens to be importable: a
    ROCm host with ``pynvml`` present (it is a pure-Python package and installs
    anywhere) would otherwise select NVML and answer every query about hardware
    that is not there.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    if torch.version.hip:
        raise RuntimeError(
            "No device-query backend for a ROCm build of torch. The queries "
            "this module exposes (device UUID, free/total memory) come from a "
            "vendor management library; the AMD counterpart of NVML is amdsmi, "
            "and no backend for it is implemented here -- its API has not been "
            "verified against an installed copy, and writing one from memory is "
            "how a wrong device index becomes a hung refit. Implement "
            "DeviceQueryBackend for amdsmi and install it with "
            "set_device_query_backend()."
        )

    _BACKEND = NvmlBackend()
    return _BACKEND


def visible_devices_entries() -> Optional[list[str]]:
    """Entries of the effective visible-device list, or ``None`` if unrestricted.

    Mirrors ``torch.cuda._parse_visible_devices``' precedence so this module and
    torch agree on which physical device a logical index names.

    Raises:
        RuntimeError: If ``HIP_VISIBLE_DEVICES`` names more devices than
            ``ROCR_VISIBLE_DEVICES`` left visible (torch refuses this too), or if
            ``ROCR_VISIBLE_DEVICES`` is set without ``HIP_VISIBLE_DEVICES``. In
            that second case torch renumbers the reduced pool from zero, so the
            environment no longer carries the physical indices a management
            library needs and no correct answer is recoverable from it.
    """
    value = os.environ.get(CUDA_VISIBLE_DEVICES)

    if torch.version.hip:
        hip = os.environ.get(HIP_VISIBLE_DEVICES)
        rocr = os.environ.get(ROCR_VISIBLE_DEVICES)
        if rocr is not None:
            rocr_entries = rocr.split(",")
            if hip is None:
                raise RuntimeError(
                    f"{ROCR_VISIBLE_DEVICES}={rocr!r} is set without "
                    f"{HIP_VISIBLE_DEVICES}. torch renumbers the reduced pool "
                    "from zero, so a logical device index no longer identifies a "
                    "physical device and the management library cannot be queried "
                    f"for the right one. Set {HIP_VISIBLE_DEVICES} as well."
                )
            if len(hip.split(",")) > len(rocr_entries):
                raise RuntimeError(
                    f"{HIP_VISIBLE_DEVICES}={hip!r} names more devices than "
                    f"{ROCR_VISIBLE_DEVICES}={rocr!r} left visible."
                )
            value = hip
        elif hip is not None:
            value = hip

    if not value:
        return None
    return value.split(",")


def device_id_to_physical_device_id(device_id: int) -> int:
    """Convert a logical torch device index to a physical (management) index.

    Args:
        device_id: Logical device index (``torch.cuda.current_device()``).

    Returns:
        Physical device index.

    Raises:
        RuntimeError: If the logical index is out of range for the visible list,
            or if a visible entry is not a plain integer (a UUID entry — both
            CUDA and HIP accept those, and neither this translation nor the
            callers below can resolve one without querying every device).
    """
    entries = visible_devices_entries()
    if entries is None:
        return device_id
    try:
        return int(entries[device_id])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to convert logical device {device_id} to a physical device "
            f"index from visible devices {entries!r}: {exc}"
        ) from exc


def get_device_uuid(device_idx: int) -> str:
    """UUID of a logical device (e.g. ``'GPU-xxxxxxxx-xxxx-...'``)."""
    physical_idx = device_id_to_physical_device_id(device_idx)
    return resolve_device_query_backend().device_uuid(physical_idx)


def get_free_memory_bytes(device_idx: int) -> int:
    """Free memory in bytes for a logical device."""
    physical_idx = device_id_to_physical_device_id(device_idx)
    return resolve_device_query_backend().memory_info(physical_idx)[0]


def get_total_memory_bytes(device_idx: int) -> int:
    """Total memory capacity in bytes for a logical device.

    Unlike :func:`get_free_memory_bytes` this reports capacity and is
    independent of what is currently allocated — the correct basis for a
    preflight memory budget, with transient buffers and the device context
    accounted for separately as a fixed overhead.
    """
    physical_idx = device_id_to_physical_device_id(device_idx)
    return resolve_device_query_backend().memory_info(physical_idx)[1]


def visible_devices_env() -> Optional[tuple[str, str]]:
    """The variable actually restricting this process, as ``(name, value)``.

    ``None`` when nothing restricts it. Propagating a visible-device setting to
    a child means writing back the *same* variable a reader would consult, so
    the name travels with the value: writing ``CUDA_VISIBLE_DEVICES`` on a host
    where ``HIP_VISIBLE_DEVICES`` is what Ray set would leave the child reading
    a value nothing else agrees with.
    """
    entries = visible_devices_entries()
    if entries is None:
        return None
    if torch.version.hip and HIP_VISIBLE_DEVICES in os.environ:
        return HIP_VISIBLE_DEVICES, ",".join(entries)
    return CUDA_VISIBLE_DEVICES, ",".join(entries)


def visible_devices_value() -> Optional[str]:
    """Effective visible-device list as a string, or ``None`` if unrestricted."""
    resolved = visible_devices_env()
    return None if resolved is None else resolved[1]


def clear_visible_devices(env: Optional[MutableMapping[str, str]] = None) -> None:
    """Drop every visible-device restriction from ``env`` (default ``os.environ``).

    Used where the caller hands device assignment to something else -- Ray, or
    vLLM's own multi-GPU placement. Clearing one vendor's variable and leaving
    another's is the failure this exists to prevent: the delegated allocator
    picks devices, and the surviving variable silently remaps them.
    """
    target = os.environ if env is None else env
    for name in VISIBLE_DEVICE_ENV_VARS:
        target.pop(name, None)


def suppress_ray_visible_devices(env_vars: MutableMapping[str, str]) -> None:
    """Tell Ray not to set the visible-device variable, for either vendor."""
    for name in NOSET_VISIBLE_DEVICE_ENV_VARS:
        env_vars[name] = "1"


def drop_ray_visible_device_opt_outs(
    env_vars: MutableMapping[str, str],
) -> None:
    """Remove Ray's opt-out flags, which a worker must set for itself."""
    for name in NOSET_VISIBLE_DEVICE_ENV_VARS:
        env_vars.pop(name, None)
