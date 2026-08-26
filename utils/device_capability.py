"""Device compute-capability resolution and precision validation.

Before this, the entire tree contained exactly one capability check --
`torch.cuda.get_device_capability()[0] >= 8` at
`models/generation/vllm/vllm_worker.py`, used only as a default for
`enable_prefix_caching`. Nothing gated a requested precision on what the target
hardware can actually run, so a config asking for `precision: fp8` on an A100
failed somewhere inside vLLM rather than at startup with a usable message.

Thresholds here are derived from the shipped binaries, not from documentation.
`vllm._custom_ops.cutlass_scaled_mm_supports_fp8` and its block/group-gemm
siblings take a capability *integer*, so the real cutoffs are checkable without
a GPU. Measured against vLLM 0.27.1:

    sm       device                fp8    block-fp8   group-gemm
    70 / 75  V100 / T4             no     no          no
    80 / 86  A100 / A10            no     no          no
    89       L40S / L4 (Ada)       YES    no          no
    90       H100 / H200           YES    YES         YES
    100/103  B200 / GB300          YES    YES         YES
    120      RTX PRO 6000          YES    YES         no

Two consequences that documentation alone would have got wrong:

- **dockyard's FP8 needs sm90, not sm89.** Plain FP8 starts at Ada (sm89), but
  dockyard only ever emits *block-scaled* FP8 -- `FP8_BLOCK_QUANT_KWARGS` pins
  `weight_block_size=[128, 128]` with no config override, and `is_fp8_model`
  asserts block scaling is present. Block FP8 starts at sm90. An L40S therefore
  passes a naive `supports_fp8` check and still cannot run this path.
- **`torch.cuda.is_bf16_supported()` must not be used to gate bf16.** It
  defaults to `including_emulation=True`: below sm80 it falls through to
  `_check_bf16_tensor_supported`, which only asks whether a bf16 tensor can be
  *created*. On a T4 it returns True while every bf16 op is emulated. Native
  bf16 is `major >= 8`, and that is what `supports_native_bf16` reports.

No MXFP8 / NVFP4 predicate ships here yet, and the omission is deliberate.
A `supports_mxfp8_nvfp4` property gating on sm100 was written alongside the
three above; measuring vLLM 0.27.1's kernel ladder showed both halves of it to
be wrong. `_POSSIBLE_MXFP8_KERNELS[CUDA]` is FlashInfer CuteDSL, FlashInfer
CUTLASS, Marlin, emulation, Humming -- and only the two FlashInfer kernels
require sm100. `MarlinMxfp8LinearKernel.is_supported` defers to
`is_fp8_marlin_supported()`, which is `has_device_capability(75)`, and Marlin
sits *above* emulation in that list, so MXFP8 linear reaches a real kernel from
sm75 up. One integer cannot express that: sm100 is the native-tensor-core line,
sm75 the Marlin-fallback line, and which one a deployment should refuse below
depends on a performance claim no CPU measurement can settle. The predicate
also had no consumer -- dockyard has no MXFP8 precision path, and a capability
question nothing asks is the shape the wiring gate exists to catch. It returns
when C2 lands a path that reads it, with the threshold that path actually needs.

Degradation policy: **refuse, never downgrade.** If a config asks for a
precision the device cannot run natively, startup fails with the device name,
its capability, and what would be required. Silently substituting a different
dtype would change training numerics without telling the operator -- the same
failure shape as every defect in Track B, where nothing errored and the run
looked healthy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Capability cutoffs as sm numbers (major * 10 + minor). See the module
# docstring for how these were measured.
SM_NATIVE_BF16 = 80
SM_FP8 = 89
SM_BLOCK_FP8 = 90
SM_PREFIX_CACHING = 80


@dataclass(frozen=True)
class DeviceCapability:
    """What a single CUDA device can run natively.

    Constructed directly in tests; `resolve_device_capability` builds it from
    the live device.
    """

    major: int
    minor: int
    name: str = "unknown device"

    @property
    def sm(self) -> int:
        """Capability as the integer vLLM's predicates take (e.g. 90)."""
        return self.major * 10 + self.minor

    @property
    def supports_native_bf16(self) -> bool:
        """Native bfloat16, i.e. Ampere or newer.

        Deliberately not torch.cuda.is_bf16_supported(), which counts emulation
        and returns True on a T4.
        """
        return self.sm >= SM_NATIVE_BF16

    @property
    def supports_fp8(self) -> bool:
        """Plain FP8 scaled_mm (Ada sm89 and newer)."""
        return self.sm >= SM_FP8

    @property
    def supports_block_fp8(self) -> bool:
        """Block-scaled FP8 (Hopper sm90 and newer) -- the only FP8 dockyard emits."""
        return self.sm >= SM_BLOCK_FP8

    @property
    def supports_prefix_caching(self) -> bool:
        """Matches the pre-existing vllm_worker default this module replaces."""
        return self.sm >= SM_PREFIX_CACHING

    def describe(self) -> str:
        return f"{self.name} (compute capability {self.major}.{self.minor}, sm{self.sm})"


def resolve_device_capability(
    device: Optional[int] = None,
) -> Optional[DeviceCapability]:
    """Read the current device's capability, or None when there is no CUDA device.

    None is a legitimate answer, not an error: the sandbox fleet is CPU-only and
    the unit suite runs without a GPU. Callers treat None as "cannot validate"
    and skip, rather than refusing to start.

    Args:
        device: CUDA device index; defaults to the current device.

    Returns:
        The device's capability, or None if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability(device)
        name = torch.cuda.get_device_name(device)
    except Exception as exc:  # pragma: no cover - needs a broken CUDA install
        logger.warning("Could not read device capability: %s", exc)
        return None
    return DeviceCapability(major=major, minor=minor, name=name)


# Precision -> (predicate name, human requirement). Keyed by the strings the
# configs actually use: policy.precision and policy.generation.vllm_cfg.precision.
_PRECISION_REQUIREMENTS: dict[str, tuple[str, str]] = {
    "bfloat16": ("supports_native_bf16", "Ampere (sm80) or newer"),
    "fp8": ("supports_block_fp8", "Hopper (sm90) or newer, for block-scaled FP8"),
}


def validate_precision(
    precision: str,
    capability: Optional[DeviceCapability],
    *,
    context: str,
) -> None:
    """Refuse a precision the device cannot run natively.

    Args:
        precision: The configured precision string.
        capability: The device's capability, or None to skip (no CUDA).
        context: Where the precision came from, for the message
            (e.g. "policy.precision").

    Raises:
        ValueError: The device cannot run this precision natively.
    """
    if capability is None:
        logger.info(
            "No CUDA device visible; skipping the hardware check for "
            "%s=%r. It will be re-checked on the worker that owns a GPU.",
            context,
            precision,
        )
        return

    requirement = _PRECISION_REQUIREMENTS.get(precision)
    if requirement is None:
        # float32 / float16 run everywhere CUDA 13 still supports; an unknown
        # string is the caller's own validation problem, not this module's.
        return

    predicate, needed = requirement
    if getattr(capability, predicate):
        return

    extra = ""
    if precision == "fp8" and capability.supports_fp8:
        extra = (
            " This device supports plain FP8 but not block-scaled FP8, and "
            "dockyard only emits block-scaled FP8 (weight_block_size is pinned "
            "to [128, 128] in FP8_BLOCK_QUANT_KWARGS)."
        )
    elif precision == "bfloat16":
        extra = (
            " Note that torch.cuda.is_bf16_supported() would report True here "
            "because it counts emulation; emulated bf16 is not usable for "
            "training throughput."
        )

    raise ValueError(
        f"{context}={precision!r} requires {needed}, but this run is on "
        f"{capability.describe()}.{extra} Set {context} to a precision this "
        f"device supports, or run on newer hardware. dockyard refuses rather "
        f"than substituting a different dtype, because a silent downgrade "
        f"changes training numerics without reporting it."
    )
