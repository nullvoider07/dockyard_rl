"""Post-training GGUF export for finished dockyard_rl models.

After training completes, ``convert_dcp_to_hf`` (utils/native_checkpoint.py) produces a
regular Hugging Face checkpoint. This module converts that HF checkpoint to a GGUF file
so the released model also runs under llama.cpp / ollama and other native GGUF runtimes
— a switchable release artifact alongside the HF build.

Security note (GHSA-5jv2-g5wq-cmr4): the vulnerable component is vLLM's GGUF dequant
kernel. This export path never invokes it. Conversion shells out to llama.cpp tooling
and validation reads the file with the ``gguf`` package, so producing and checking a
GGUF here does not exercise the vLLM kernel. Serving a GGUF in vLLM is a separate,
explicitly hardened path; keep it off the export flow to leave #51 unexercised.

llama.cpp tooling (``convert_hf_to_gguf.py`` and the ``llama-quantize`` binary) is a
build-time dependency located via ``llama_cpp_dir`` or ``DOCKYARD_LLAMA_CPP_DIR``; it is
not a Python package and is not imported. ``sys.executable`` runs the converter (no
runtime venv).
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

# Quant types the llama.cpp HF converter emits directly via --outtype. Anything else
# (Q4_K_M, Q5_K_M, Q8_0, ...) is produced by quantizing an f16 GGUF with llama-quantize.
_DIRECT_OUTTYPES = {"f32", "f16", "bf16"}

# Candidate relative locations of the llama-quantize binary inside a llama.cpp checkout.
_QUANTIZE_REL_PATHS = ("llama-quantize", "build/bin/llama-quantize", "bin/llama-quantize")


def _resolve_llama_cpp_dir(llama_cpp_dir: Optional[str]) -> str:
    d = llama_cpp_dir or os.environ.get("DOCKYARD_LLAMA_CPP_DIR")
    if not d:
        raise FileNotFoundError(
            "llama.cpp tooling not located. Pass llama_cpp_dir or set "
            "DOCKYARD_LLAMA_CPP_DIR to a llama.cpp checkout providing "
            "convert_hf_to_gguf.py and the llama-quantize binary."
        )
    if not os.path.isfile(os.path.join(d, "convert_hf_to_gguf.py")):
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found under {d}.")
    return d


def _find_quantize_bin(llama_cpp_dir: str) -> str:
    for rel in _QUANTIZE_REL_PATHS:
        candidate = os.path.join(llama_cpp_dir, rel)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        f"llama-quantize binary not found under {llama_cpp_dir} "
        f"(looked in {', '.join(_QUANTIZE_REL_PATHS)}). Build llama.cpp first."
    )


def convert_hf_to_gguf(
    hf_ckpt_path: str,
    gguf_out_path: str,
    quant_type: str = "f16",
    llama_cpp_dir: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Convert a Hugging Face checkpoint directory to a GGUF file.

    Runs llama.cpp's ``convert_hf_to_gguf.py`` to emit an unquantized GGUF, then — for a
    quantized ``quant_type`` — ``llama-quantize``. The result is validated with the
    ``gguf`` package (no vLLM), so the GGUF dequant advisory is not exercised here.

    Args:
        hf_ckpt_path:   HF checkpoint directory (e.g. the output of convert_dcp_to_hf).
        gguf_out_path:  Destination ``.gguf`` path.
        quant_type:     ``f16``/``bf16``/``f32`` (emitted directly) or a llama.cpp
                        quant name such as ``Q4_K_M``/``Q8_0`` (via llama-quantize).
        llama_cpp_dir:  llama.cpp checkout; falls back to ``DOCKYARD_LLAMA_CPP_DIR``.
        overwrite:      Overwrite an existing output file.

    Returns:
        The path to the written GGUF file.
    """
    if not os.path.isdir(hf_ckpt_path):
        raise FileNotFoundError(f"HF checkpoint directory not found: {hf_ckpt_path}")
    if os.path.exists(gguf_out_path) and not overwrite:
        raise FileExistsError(
            f"GGUF already exists at {gguf_out_path}. Delete it or set overwrite=True."
        )
    llama_cpp_dir = _resolve_llama_cpp_dir(llama_cpp_dir)
    os.makedirs(os.path.dirname(os.path.abspath(gguf_out_path)) or ".", exist_ok=True)

    converter = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    qt = quant_type.lower()

    if qt in _DIRECT_OUTTYPES:
        subprocess.run(
            [sys.executable, converter, hf_ckpt_path,
             "--outfile", gguf_out_path, "--outtype", qt],
            check=True,
        )
    else:
        tmp_f16 = gguf_out_path + ".f16.gguf"
        try:
            subprocess.run(
                [sys.executable, converter, hf_ckpt_path,
                 "--outfile", tmp_f16, "--outtype", "f16"],
                check=True,
            )
            quantize_bin = _find_quantize_bin(llama_cpp_dir)
            subprocess.run([quantize_bin, tmp_f16, gguf_out_path, quant_type], check=True)
        finally:
            if os.path.exists(tmp_f16):
                os.remove(tmp_f16)

    validate_gguf(gguf_out_path)
    return gguf_out_path


def validate_gguf(gguf_path: str, *, max_float_tensors_checked: int = 8) -> dict:
    """Validate a GGUF file without vLLM (leaves GHSA-5jv2-g5wq-cmr4 unexercised).

    Reads the file with the ``gguf`` package: confirms ``general.architecture`` and at
    least one tensor are present, and that a sample of float (unquantized) tensors are
    finite. Quantized tensor blocks are stored raw and are not decoded here.

    Returns a small metadata summary.
    """
    import numpy as np
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    if not reader.tensors:
        raise ValueError(f"GGUF {gguf_path} contains no tensors.")
    if reader.get_field("general.architecture") is None:
        raise ValueError(f"GGUF {gguf_path} is missing general.architecture.")

    checked = 0
    for tensor in reader.tensors:
        arr = np.asarray(tensor.data)
        if arr.dtype.kind != "f":
            continue
        if not np.isfinite(arr).all():
            raise ValueError(
                f"GGUF tensor {tensor.name} in {gguf_path} contains non-finite values."
            )
        checked += 1
        if checked >= max_float_tensors_checked:
            break

    return {
        "path": gguf_path,
        "n_tensors": len(reader.tensors),
        "float_tensors_checked": checked,
    }


def export_gguf_variants(
    hf_ckpt_path: str,
    out_dir: str,
    quant_types: Optional[list[str]] = None,
    model_basename: str = "model",
    llama_cpp_dir: Optional[str] = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """Produce one or more GGUF builds from an HF checkpoint (switchable release).

    Each requested ``quant_type`` yields ``<out_dir>/<model_basename>.<quant_type>.gguf``.
    Returns a map of quant_type -> output path.
    """
    quant_types = quant_types or ["f16", "Q4_K_M"]
    os.makedirs(out_dir, exist_ok=True)
    outputs: dict[str, str] = {}
    for quant_type in quant_types:
        out_path = os.path.join(out_dir, f"{model_basename}.{quant_type}.gguf")
        outputs[quant_type] = convert_hf_to_gguf(
            hf_ckpt_path,
            out_path,
            quant_type=quant_type,
            llama_cpp_dir=llama_cpp_dir,
            overwrite=overwrite,
        )
    return outputs
