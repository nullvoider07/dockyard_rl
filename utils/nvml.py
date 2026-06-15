"""NVIDIA Management Library (NVML) helpers for Project Dockyard.

Provides GPU UUID and free-memory queries that are CUDA_VISIBLE_DEVICES-aware,
resolving logical device indices to physical device indices before calling
into pynvml.

pynvml is installed as part of the nvidia-ml-py package which ships with
every CUDA-enabled environment including ubuntu-swe.
"""

import contextlib
import os
from typing import Generator
import pynvml

@contextlib.contextmanager
def nvml_context() -> Generator[None, None, None]:
    """Context manager that initialises and shuts down NVML cleanly.

    Raises:
        RuntimeError: If NVML initialisation fails (e.g. no NVIDIA driver).
    """
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

def device_id_to_physical_device_id(device_id: int) -> int:
    """Convert a logical CUDA device index to a physical device index.

    When CUDA_VISIBLE_DEVICES is set, PyTorch's logical device 0 maps to
    the first entry in the env var, not necessarily physical GPU 0.
    NVML always uses physical indices, so this translation is required
    before calling any nvmlDevice* function.

    Args:
        device_id: Logical CUDA device index (torch.cuda.current_device()).

    Returns:
        Physical NVML device index.

    Raises:
        RuntimeError: If the logical index is out of range for the visible
                      device list, or if a visible device entry is not a
                      plain integer (e.g. a MIG UUID — not supported here).
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        entries = cvd.split(",")
        try:
            return int(entries[device_id])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to convert logical device {device_id} to a physical "
                f"device index. CUDA_VISIBLE_DEVICES={cvd!r}: {exc}"
            ) from exc
    return device_id

def get_device_uuid(device_idx: int) -> str:
    """Return the UUID of a CUDA device (e.g. 'GPU-xxxxxxxx-xxxx-...').

    Args:
        device_idx: Logical CUDA device index.

    Returns:
        UUID string in the format 'GPU-<hex>'.

    Raises:
        RuntimeError: On NVML error or unexpected UUID type.
    """
    physical_idx = device_id_to_physical_device_id(device_idx)
    with nvml_context():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)
            uuid   = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                return uuid.decode("utf-8")
            if isinstance(uuid, str):
                return uuid
            raise RuntimeError(
                f"Unexpected UUID type {type(uuid)} for device {device_idx} "
                f"(physical index {physical_idx})."
            )
        except pynvml.NVMLError as exc:
            raise RuntimeError(
                f"Failed to get UUID for device {device_idx} "
                f"(physical index {physical_idx}): {exc}"
            ) from exc

def get_free_memory_bytes(device_idx: int) -> int:
    """Return the free GPU memory in bytes for a CUDA device.

    Args:
        device_idx: Logical CUDA device index.

    Returns:
        Free memory in bytes.

    Raises:
        RuntimeError: On NVML error.
    """
    physical_idx = device_id_to_physical_device_id(device_idx)
    with nvml_context():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)
            return int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
        except pynvml.NVMLError as exc:
            raise RuntimeError(
                f"Failed to get free memory for device {device_idx} "
                f"(physical index {physical_idx}): {exc}"
            ) from exc