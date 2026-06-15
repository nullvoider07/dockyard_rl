"""Unit tests for the GDPval agentic file-producing path (CPU-only).

Covers the standalone in-container extractor, the environment's pure command
builders + grading hooks (sandbox client + reward stubbed), the dataset's agentic
format_data (and text-path byte-identity), reference-file encoding, and the
agentic data processor's metadata routing. Live container file production and the
doc-library extraction of real xlsx/docx/pptx/pdf are hardware/image-gated.
"""

import base64
import json

import pytest

from dockyard_rl.environments import _gdpval_extract as gx
from dockyard_rl.environments import gdpval_agentic_environment as env_mod
from dockyard_rl.environments.gdpval_agentic_environment import (
    _GDPvalAgenticEnvironment,
    _safe_member_name,
    build_extraction_command,
    build_write_file_command,
    compose_deliverable,
)
from dockyard_rl.data.datasets.gdpval import (
    GDPvalDataset,
    _AGENTIC_SYSTEM_PROMPT,
    _SYSTEM_PROMPT,
    _encode_reference_files,
)


# --------------------------------------------------------------------------- #
# Extractor (_gdpval_extract.py)                                              #
# --------------------------------------------------------------------------- #
class TestExtractor:
    def test_extract_dir_manifest_and_content(self, tmp_path):
        (tmp_path / "report.md").write_text("# Title\nbody text", encoding="utf-8")
        (tmp_path / "data.csv").write_text("a,b\n1,2", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "notes.txt").write_text("hello", encoding="utf-8")

        out = gx.extract_dir(str(tmp_path))
        assert "=== DELIVERABLE MANIFEST ===" in out
        assert "=== EXTRACTED CONTENT ===" in out
        # All files listed (relative paths) and content present.
        assert "report.md" in out
        assert "data.csv" in out
        assert "sub/notes.txt" in out
        assert "body text" in out
        assert "hello" in out

    def test_extract_dir_empty(self, tmp_path):
        out = gx.extract_dir(str(tmp_path))
        assert "empty" in out.lower()

    def test_extract_dir_missing(self, tmp_path):
        out = gx.extract_dir(str(tmp_path / "nope"))
        assert "no deliverable directory" in out.lower()

    def test_extract_dir_char_limit_truncates(self, tmp_path):
        (tmp_path / "big.txt").write_text("x" * 5000, encoding="utf-8")
        out = gx.extract_dir(str(tmp_path), char_limit=500)
        assert len(out) <= 500 + len("\n... [truncated]")
        assert "truncated" in out

    def test_extract_file_text(self, tmp_path):
        p = tmp_path / "a.json"
        p.write_text('{"k": 1}', encoding="utf-8")
        assert '{"k": 1}' in gx.extract_file(str(p))

    def test_extract_file_unknown_ext_fallback(self, tmp_path):
        p = tmp_path / "thing.bin"
        p.write_bytes(b"\x00\x01\x02")
        out = gx.extract_file(str(p))
        assert "no text extractor" in out

    def test_extract_file_binary_missing_lib_degrades(self, tmp_path):
        # A .xlsx that is not a real workbook: extractor must not raise.
        p = tmp_path / "broken.xlsx"
        p.write_bytes(b"not a workbook")
        out = gx.extract_file(str(p))
        assert out.startswith("[")  # placeholder, not an exception

    def test_walk_files_sorted_relative(self, tmp_path):
        (tmp_path / "b.txt").write_text("", encoding="utf-8")
        (tmp_path / "a.txt").write_text("", encoding="utf-8")
        rels = [r for r, _ap, _s in gx.walk_files(str(tmp_path))]
        assert rels == ["a.txt", "b.txt"]


# --------------------------------------------------------------------------- #
# Environment pure helpers                                                    #
# --------------------------------------------------------------------------- #
class TestEnvHelpers:
    def test_build_extraction_command_roundtrips_source(self):
        src = "print('hi')\n"
        cmd = build_extraction_command("/workspace/deliverable", 1234, extractor_source=src)
        b64 = base64.b64encode(src.encode()).decode()
        assert b64 in cmd
        assert "base64 -d" in cmd
        assert "/workspace/deliverable" in cmd
        assert "1234" in cmd

    def test_build_write_file_command(self):
        cmd = build_write_file_command("/workspace/reference", "input.xlsx", "QUJD")
        assert "mkdir -p" in cmd
        assert "base64 -d" in cmd
        assert "input.xlsx" in cmd
        assert "QUJD" in cmd

    def test_safe_member_name_strips_traversal(self):
        assert _safe_member_name("../../etc/passwd") == "passwd"
        assert _safe_member_name("a/b/c.txt") == "c.txt"
        assert _safe_member_name("....") == "ref"
        assert _safe_member_name("normal.docx") == "normal.docx"

    def test_compose_deliverable_prepends_header(self):
        out = compose_deliverable("=== DELIVERABLE MANIFEST ===\n- a.md")
        assert out.startswith("The following is an automated extraction")
        assert "MANIFEST" in out

    def test_compose_deliverable_empty_is_empty(self):
        assert compose_deliverable("") == ""
        assert compose_deliverable("   \n ") == ""


# --------------------------------------------------------------------------- #
# Environment grading hooks (sandbox client + reward stubbed)                 #
# --------------------------------------------------------------------------- #
class _StubReward:
    def __init__(self, reward, status="ok", failure_reason=None):
        self._r = type("R", (), {
            "reward": reward, "status": status, "failure_reason": failure_reason,
        })()
        self.last_traj: dict | None = None

    def __call__(self, traj):
        self.last_traj = traj
        return self._r


def _make_env(reward_stub):
    cfg = {
        "sandbox_urls": ["http://localhost:9090"],
        "reward_mode": "weighted",
        "deliverable_dir": "/workspace/deliverable",
        "reference_dir": "/workspace/reference",
        "image": "ubuntu-swe-gdpval:latest",
        "output_limit": 4096,
        "extraction_char_limit": 50000,
    }
    env = _GDPvalAgenticEnvironment(cfg)
    env._reward = reward_stub
    return env


class TestEnvHooks:
    def test_start_spec_image_mode(self):
        env = _make_env(_StubReward(1.0))
        spec = env._start_spec({"image": "custom:tag"})
        assert spec.get("mode") == "image"
        assert spec.get("image") == "custom:tag"
        assert spec.get("block_network") is False

    def test_start_spec_default_image(self):
        env = _make_env(_StubReward(1.0))
        spec = env._start_spec({})
        assert spec.get("image") == "ubuntu-swe-gdpval:latest"

    def test_after_start_provisions_reference_files(self, monkeypatch):
        env = _make_env(_StubReward(1.0))
        calls = []
        monkeypatch.setattr(
            env_mod, "session_exec",
            lambda url, sid, cmd, **kw: calls.append(cmd) or {"stdout": "", "exit_code": 0},
        )
        env._after_start("http://x", "sid", {
            "reference_files": {"a.xlsx": "QUJD", "b.csv": "REVG"},
        })
        assert len(calls) == 2
        assert any("a.xlsx" in c for c in calls)
        assert any("b.csv" in c for c in calls)

    def test_after_start_noop_without_refs(self, monkeypatch):
        env = _make_env(_StubReward(1.0))
        calls = []
        monkeypatch.setattr(
            env_mod, "session_exec",
            lambda *a, **k: calls.append(1) or {"stdout": ""},
        )
        env._after_start("http://x", "sid", {})
        assert calls == []

    def test_finish_and_score_grades_extracted_text(self, monkeypatch):
        reward = _StubReward(0.75, status="ok", failure_reason="rubric: 3/4")
        env = _make_env(reward)
        monkeypatch.setattr(
            env_mod, "session_exec",
            lambda *a, **k: {"stdout": "=== DELIVERABLE MANIFEST ===\n- out.xlsx", "exit_code": 0},
        )
        r, verdict = env._finish_and_score("http://x", "sid", {
            "prompt": "build a workbook", "rubric_json": "[]",
        })
        assert r == 0.75
        assert "rubric" in verdict
        # Reward saw the composed deliverable (header + manifest) + task fields.
        assert reward.last_traj is not None
        assert reward.last_traj["deliverable"].startswith("The following is an automated")
        assert "MANIFEST" in reward.last_traj["deliverable"]
        assert reward.last_traj["prompt"] == "build a workbook"

    def test_finish_and_score_no_files_scores_zero(self, monkeypatch):
        reward = _StubReward(1.0)
        env = _make_env(reward)
        monkeypatch.setattr(env_mod, "session_exec", lambda *a, **k: {"stdout": "   "})
        r, verdict = env._finish_and_score("http://x", "sid", {"prompt": "p", "rubric_json": "[]"})
        assert r == 0.0
        assert "no deliverable" in verdict.lower()
        assert reward.last_traj is None  # reward not invoked

    def test_finish_and_score_extraction_error_scores_zero(self, monkeypatch):
        env = _make_env(_StubReward(1.0))

        def _boom(*a, **k):
            raise RuntimeError("exec failed")

        monkeypatch.setattr(env_mod, "session_exec", _boom)
        r, verdict = env._finish_and_score("http://x", "sid", {"prompt": "p", "rubric_json": "[]"})
        assert r == 0.0
        assert "extraction error" in verdict.lower()


# --------------------------------------------------------------------------- #
# Dataset format_data (agentic) + text-path byte-identity                     #
# --------------------------------------------------------------------------- #
def _make_ds(agentic):
    ds = GDPvalDataset.__new__(GDPvalDataset)
    ds.task_name = "gdpval"
    ds.agentic = agentic
    ds.image = "ubuntu-swe-gdpval:latest"
    ds.deliverable_dir = "/workspace/deliverable"
    ds.reference_dir = "/workspace/reference"
    ds.max_turns = 24
    ds.exec_timeout_sec = 180
    ds.provision_reference_files = False
    return ds


class TestDatasetFormat:
    _ROW: dict[str, object] = {"prompt": "Do the task", "rubric_json": "[]",
                               "task_id": "t1", "occupation": "analyst",
                               "sector": "finance"}

    def test_text_path_schema_unchanged(self):
        ds = _make_ds(agentic=False)
        out = ds.format_data(dict(self._ROW))
        assert set(out.keys()) == {
            "messages", "task_name", "task_id", "occupation", "sector",
            "prompt", "rubric_json",
        }
        assert out["messages"][0]["content"] == _SYSTEM_PROMPT

    def test_agentic_schema_and_system_prompt(self):
        ds = _make_ds(agentic=True)
        out = ds.format_data(dict(self._ROW))
        assert out["image"] == "ubuntu-swe-gdpval:latest"
        assert out["deliverable_dir"] == "/workspace/deliverable"
        assert out["max_turns"] == 24
        assert out["exec_timeout_sec"] == 180
        assert json.loads(out["reference_files_json"]) == {}
        sys_prompt = out["messages"][0]["content"]
        assert "/workspace/deliverable" in sys_prompt
        assert "TASK_COMPLETE" in sys_prompt
        assert "```bash" in sys_prompt

    def test_agentic_system_prompt_formats_dirs(self):
        assert "{deliverable_dir}" in _AGENTIC_SYSTEM_PROMPT  # template has the slot
        ds = _make_ds(agentic=True)
        out = ds.format_data(dict(self._ROW))
        assert "{deliverable_dir}" not in out["messages"][0]["content"]

    def test_agentic_provisions_reference_files_when_enabled(self):
        ds = _make_ds(agentic=True)
        ds.provision_reference_files = True
        row = dict(self._ROW)
        row["reference_files"] = [{"filename": "in.xlsx", "bytes": b"ABC"}]
        out = ds.format_data(row)
        refs = json.loads(out["reference_files_json"])
        assert refs["in.xlsx"] == base64.b64encode(b"ABC").decode()


class TestEncodeReferenceFiles:
    def test_dict_with_bytes(self):
        out = _encode_reference_files({"reference_files": [{"name": "a.bin", "bytes": b"\x00\x01"}]})
        assert out["a.bin"] == base64.b64encode(b"\x00\x01").decode()

    def test_raw_bytes_entry_gets_synthetic_name(self):
        out = _encode_reference_files({"reference_files": [b"hello"]})
        assert out["reference_0"] == base64.b64encode(b"hello").decode()

    def test_uri_only_skipped(self):
        out = _encode_reference_files({"reference_files": [{"path": "s3://x/y.pdf"}]})
        assert out == {}

    def test_empty(self):
        assert _encode_reference_files({}) == {}
        assert _encode_reference_files({"reference_files": None}) == {}


# --------------------------------------------------------------------------- #
# Processor routing                                                           #
# --------------------------------------------------------------------------- #
class TestProcessor:
    def test_routes_session_metadata(self, monkeypatch):
        import dockyard_rl.data.processors as proc

        monkeypatch.setattr(
            proc, "get_formatted_message_log",
            lambda *a, **k: [{"role": "user", "content": "x", "token_ids": [1, 2, 3]}],
        )
        datum = {
            "messages": [{"role": "user", "content": "x"}],
            "task_name": "gdpval",
            "task_id": "t1",
            "occupation": "analyst",
            "sector": "finance",
            "prompt": "do it",
            "rubric_json": "[]",
            "image": "img:tag",
            "deliverable_dir": "/workspace/deliverable",
            "reference_dir": "/workspace/reference",
            "max_turns": 24,
            "exec_timeout_sec": 180,
            "reference_files_json": json.dumps({"a.xlsx": "QUJD"}),
        }
        out = proc.gdpval_agentic_data_processor(datum, None, None, None, 0)  # type: ignore[arg-type]
        info = out["extra_env_info"]
        assert info is not None
        assert info["image"] == "img:tag"
        assert info["deliverable_dir"] == "/workspace/deliverable"
        assert info["max_turns"] == 24
        assert info["exec_timeout_sec"] == 180
        assert info["prompt"] == "do it"
        assert info["reference_files"] == {"a.xlsx": "QUJD"}
        assert out.get("task_name") == "gdpval"
        assert out["loss_multiplier"] == 1.0


# --------------------------------------------------------------------------- #
# Registry wiring                                                             #
# --------------------------------------------------------------------------- #
class TestWiring:
    def test_env_registered_and_resolves(self):
        import importlib

        from dockyard_rl.environments.utils import ENV_REGISTRY

        entry = ENV_REGISTRY["gdpval_agentic"]
        assert entry.get("default_processor") == "gdpval_agentic_data_processor"
        mod, _, cls = entry["actor_class_fqn"].rpartition(".")
        actor = getattr(importlib.import_module(mod), cls)
        assert actor is not None

    def test_processor_registered(self):
        from dockyard_rl.data.processors import PROCESSOR_REGISTRY

        assert "gdpval_agentic_data_processor" in PROCESSOR_REGISTRY

    def test_dataset_registered(self):
        from dockyard_rl.data.datasets.response_datasets import DATASET_REGISTRY

        assert "gdpval_agentic" in DATASET_REGISTRY


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
