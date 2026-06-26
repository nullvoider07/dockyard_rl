"""Post-training GGUF export (utils/gguf_export.py).

Validation round-trips a real GGUF via the ``gguf`` package — never vLLM — so the GGUF
dequant advisory (GHSA-5jv2-g5wq-cmr4) is not exercised. The llama.cpp conversion is
covered by mocking subprocess (the binaries are a build-time dependency, GPU/deploy
only); the command construction and the f16→quantize two-step are asserted.

Imports resolve via tests/conftest.py (repo parent on sys.path).
"""

from __future__ import annotations

import numpy as np
import pytest

from dockyard_rl.utils import gguf_export


def _write_tiny_gguf(path, arch: str = "llama", tensor=None) -> None:
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), arch)
    writer.add_architecture()
    if tensor is None:
        tensor = np.ones((2, 3), dtype=np.float32)
    writer.add_tensor("w0", tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class TestValidateGguf:
    def test_valid_file_summary(self, tmp_path):
        p = tmp_path / "model.gguf"
        _write_tiny_gguf(p)
        summary = gguf_export.validate_gguf(str(p))
        assert summary["n_tensors"] == 1
        assert summary["float_tensors_checked"] == 1

    def test_rejects_nonfinite_tensor(self, tmp_path):
        bad = np.array([[1.0, np.inf, 2.0]], dtype=np.float32)
        p = tmp_path / "bad.gguf"
        _write_tiny_gguf(p, tensor=bad)
        with pytest.raises(ValueError, match="non-finite"):
            gguf_export.validate_gguf(str(p))


class TestConvertHfToGguf:
    def test_direct_outtype_single_call(self, tmp_path, monkeypatch):
        hf = tmp_path / "hf"
        hf.mkdir()
        out = tmp_path / "model.f16.gguf"
        calls = []
        monkeypatch.setattr(gguf_export, "_resolve_llama_cpp_dir", lambda d: "/fake/llama")
        monkeypatch.setattr(gguf_export, "validate_gguf", lambda p, **k: {"path": p})
        monkeypatch.setattr(
            gguf_export.subprocess, "run", lambda cmd, **k: calls.append(cmd)
        )

        res = gguf_export.convert_hf_to_gguf(str(hf), str(out), quant_type="f16")

        assert res == str(out)
        assert len(calls) == 1
        assert calls[0][1].endswith("convert_hf_to_gguf.py")
        assert "--outtype" in calls[0] and "f16" in calls[0]

    def test_quantized_two_step(self, tmp_path, monkeypatch):
        hf = tmp_path / "hf"
        hf.mkdir()
        out = tmp_path / "model.Q4_K_M.gguf"
        calls = []
        monkeypatch.setattr(gguf_export, "_resolve_llama_cpp_dir", lambda d: "/fake/llama")
        monkeypatch.setattr(
            gguf_export, "_find_quantize_bin", lambda d: "/fake/llama/llama-quantize"
        )
        monkeypatch.setattr(gguf_export, "validate_gguf", lambda p, **k: {"path": p})
        monkeypatch.setattr(
            gguf_export.subprocess, "run", lambda cmd, **k: calls.append(cmd)
        )

        gguf_export.convert_hf_to_gguf(str(hf), str(out), quant_type="Q4_K_M")

        assert len(calls) == 2
        assert calls[0][-1] == "f16"  # convert to f16 first
        assert calls[1][0] == "/fake/llama/llama-quantize"
        assert calls[1][-1] == "Q4_K_M"  # then quantize to the requested type

    def test_missing_tooling_raises(self, tmp_path, monkeypatch):
        hf = tmp_path / "hf"
        hf.mkdir()
        monkeypatch.delenv("DOCKYARD_LLAMA_CPP_DIR", raising=False)
        with pytest.raises(FileNotFoundError, match="llama.cpp"):
            gguf_export.convert_hf_to_gguf(str(hf), str(tmp_path / "m.gguf"))

    def test_refuses_overwrite(self, tmp_path, monkeypatch):
        hf = tmp_path / "hf"
        hf.mkdir()
        out = tmp_path / "model.f16.gguf"
        out.write_bytes(b"existing")
        monkeypatch.setattr(gguf_export, "_resolve_llama_cpp_dir", lambda d: "/fake/llama")
        with pytest.raises(FileExistsError):
            gguf_export.convert_hf_to_gguf(str(hf), str(out), quant_type="f16")
