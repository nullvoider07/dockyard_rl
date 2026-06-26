"""Defense-in-depth for vLLM's GGUF dequant kernel (GHSA-5jv2-g5wq-cmr4).

vLLM's GGUF dequant CUDA kernel truncates output tensor dimensions to ``int`` and
returns a ``torch::empty`` output, so for dimensions exceeding 32-bit range the
untouched tail keeps uninitialized GPU memory. dockyard serves single-tenant (one job
owns the GPU), which removes the cross-tenant disclosure precondition; this guard adds
a boundary check that rejects out-of-range dequant dimensions before the kernel runs,
so the truncation cannot be triggered.

Mirrors quantization/fp8.py: a monkey-patch installed once, idempotently. ``gguf.py``
in vLLM calls ``ops.ggml_dequantize`` via ``from vllm import _custom_ops as ops``, so
patching the module attribute is seen at call time. This guard belongs only on the
GGUF serve path; the post-training export path (utils/gguf_export.py) never loads GGUF
in vLLM and does not need it.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

# The dequant kernel indexes its output with 32-bit ints; dimensions at or beyond this
# bound are where the truncation that leaves uninitialized memory begins.
_INT32_MAX = 2**31 - 1

_guard_installed = False


def install_gguf_dequant_guard() -> None:
    """Wrap vLLM's ``ggml_dequantize`` to reject out-of-range output dimensions.

    Idempotent; a no-op if vLLM is not importable. Call once before serving a GGUF
    model in vLLM (alongside engine init) — not on the export path.
    """
    global _guard_installed
    if _guard_installed:
        return
    try:
        from vllm import _custom_ops as ops
    except ImportError:
        return

    original = ops.ggml_dequantize

    @functools.wraps(original)
    def _guarded_ggml_dequantize(W, quant_type, m, n, dtype=None):
        for name, dim in (("m", m), ("n", n)):
            if not isinstance(dim, int) or dim <= 0 or dim > _INT32_MAX:
                raise ValueError(
                    f"GGUF dequant {name}={dim!r} is outside the validated range "
                    f"(0, {_INT32_MAX}]. Rejected to prevent the int-truncation / "
                    "uninitialized-GPU-memory leak in vLLM's dequant kernel "
                    "(GHSA-5jv2-g5wq-cmr4)."
                )
        return original(W, quant_type, m, n, dtype)

    ops.ggml_dequantize = _guarded_ggml_dequantize
    _guard_installed = True
    logger.info("Installed GGUF dequant dimension guard (GHSA-5jv2-g5wq-cmr4).")
