"""Unit tests for the structured tool-use parsing mode in the two generic envs (S5).

Pure CPU coverage of the parse/dispatch decision and the #2656 ``invalid_action``
stamping for ``MultiTurnSessionEnvironment`` (run_shell/read_file/write_file/
task_complete) and ``CodeEnvironment`` (submit_patch). The legacy fenced/diff
parsers are asserted byte-identical when the ``structured_tools`` flag is off.

These tests never touch a real sandbox: the session env's executor calls are
light-stubbed via monkeypatch so the dispatch decision (which command is exec'd,
which turn is graded, what gets stamped invalid) is observable on CPU.
"""

import base64
import json

import pytest

import dockyard_rl.environments.multi_turn_session_env as mtse
from dockyard_rl.environments.code_environment import (
    _extract_patch,
    _extract_patch_structured,
)
from dockyard_rl.environments.multi_turn_session_env import (
    MultiTurnSessionEnvironment,
    _parse_action,
    build_read_file_command,
    build_session_write_command,
)
from dockyard_rl.tool_protocol import HERMES_TOOL_CALL_START


def _call(name: str, args: dict) -> str:
    return (
        HERMES_TOOL_CALL_START
        + "\n"
        + json.dumps({"name": name, "arguments": args})
        + "\n</tool_call>"
    )


# ── a session env whose sandbox calls are stubbed ───────────────────


class _StubSessionEnv(MultiTurnSessionEnvironment):
    """Session env with no-op subclass hooks and a recording grader."""

    def _setup(self, cfg: dict) -> None:
        self.scored = False

    def _start_spec(self, meta: dict):  # type: ignore[override]
        return {"mode": "bare"}

    def _finish_and_score(self, url, session_id, meta):  # type: ignore[override]
        self.scored = True
        return 1.0, "[graded]"


@pytest.fixture
def env(monkeypatch):
    """A structured-tools session env with the executor API stubbed.

    ``exec_calls`` captures every command handed to ``session_exec`` so the
    dispatch decision (run_shell vs. cat vs. base64-write) is assertable.
    """
    exec_calls: list[str] = []

    monkeypatch.setattr(mtse, "start_session", lambda *a, **k: "sid-1")
    monkeypatch.setattr(mtse, "delete_session", lambda *a, **k: None)

    def _fake_exec(url, sid, command, **kwargs):
        exec_calls.append(command)
        return {"stdout": "OUT", "stderr": "", "exit_code": 0, "timed_out": False}

    monkeypatch.setattr(mtse, "session_exec", _fake_exec)

    e = _StubSessionEnv(
        {"sandbox_urls": ["http://x"], "structured_tools": True, "max_turns": 30}
    )
    e.exec_calls = exec_calls  # type: ignore[attr-defined]
    return e


# ── command builders ────────────────────────────────────────────────


class TestCommandBuilders:
    def test_read_file_quotes_and_dashdash(self):
        cmd = build_read_file_command("/etc/passwd")
        assert cmd == "cat -- '/etc/passwd'"

    def test_read_file_quotes_dash_path(self):
        # A path beginning with '-' must be read, not parsed as an option.
        cmd = build_read_file_command("-rf")
        assert "-- '-rf'" in cmd

    def test_read_file_escapes_single_quote(self):
        cmd = build_read_file_command("a'b")
        assert cmd == "cat -- 'a'\"'\"'b'"

    def test_write_file_base64_roundtrip(self):
        content = "line1\nline2\n"
        cmd = build_session_write_command("/w/out.txt", content)
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        assert b64 in cmd
        assert "mkdir -p" in cmd
        assert "base64 -d > '/w/out.txt'" in cmd
        # base64 of the exact content decodes back to it.
        assert base64.b64decode(b64).decode("utf-8") == content


# ── structured parse/dispatch (session env) ─────────────────────────


class TestSessionStructuredParse:
    def test_run_shell_dispatch(self, env):
        cmd, done, invalid = env._parse_structured_action(
            _call("run_shell", {"command": "ls -la"})
        )
        assert cmd == "ls -la" and not done and not invalid

    def test_read_file_dispatch(self, env):
        cmd, done, invalid = env._parse_structured_action(
            _call("read_file", {"path": "/a/b.py"})
        )
        assert cmd == build_read_file_command("/a/b.py")
        assert not done and not invalid

    def test_write_file_dispatch(self, env):
        cmd, done, invalid = env._parse_structured_action(
            _call("write_file", {"path": "/a/b.txt", "content": "hi"})
        )
        assert cmd == build_session_write_command("/a/b.txt", "hi")
        assert not done and not invalid

    def test_task_complete_done(self, env):
        cmd, done, invalid = env._parse_structured_action(_call("task_complete", {}))
        assert cmd == "" and done and not invalid

    def test_no_call_invalid(self, env):
        cmd, done, invalid = env._parse_structured_action("just thinking, no action")
        assert cmd == "" and not done and invalid

    def test_unknown_tool_invalid(self, env):
        cmd, done, invalid = env._parse_structured_action(_call("rm_rf", {}))
        assert invalid and not done and cmd == ""

    def test_malformed_json_invalid(self, env):
        text = HERMES_TOOL_CALL_START + "\n{not json}\n</tool_call>"
        cmd, done, invalid = env._parse_structured_action(text)
        assert invalid and not done and cmd == ""

    def test_schema_invalid_args_invalid(self, env):
        # run_shell.command must be a string.
        cmd, done, invalid = env._parse_structured_action(
            _call("run_shell", {"command": 123})
        )
        assert invalid

    def test_missing_required_arg_invalid(self, env):
        cmd, done, invalid = env._parse_structured_action(
            _call("write_file", {"path": "/a"})
        )
        assert invalid

    def test_multiple_calls_invalid(self, env):
        # Fork 7: a single action per turn; >1 valid call is invalid.
        text = _call("run_shell", {"command": "a"}) + _call(
            "run_shell", {"command": "b"}
        )
        cmd, done, invalid = env._parse_structured_action(text)
        assert invalid and cmd == "" and not done


# ── structured _process_one: stamping + dispatch flow ───────────────


class TestSessionProcessOne:
    def test_run_shell_execs_and_not_invalid(self, env):
        reward, obs, meta, done = env._process_one(
            _call("run_shell", {"command": "echo hi"}), {}
        )
        assert env.exec_calls == ["echo hi"]
        assert meta["invalid_action"] is False
        assert not done and reward == 0.0
        assert "OUT" in obs["content"]

    def test_read_file_execs_cat(self, env):
        env._process_one(_call("read_file", {"path": "/p"}), {})
        assert env.exec_calls == [build_read_file_command("/p")]

    def test_write_file_execs_base64(self, env):
        env._process_one(_call("write_file", {"path": "/p", "content": "x"}), {})
        assert env.exec_calls == [build_session_write_command("/p", "x")]

    def test_task_complete_grades(self, env):
        reward, obs, meta, done = env._process_one(_call("task_complete", {}), {})
        assert done and reward == 1.0 and env.scored is True
        assert meta["invalid_action"] is False
        # task_complete runs no command.
        assert env.exec_calls == []

    def test_invalid_turn_stamped_and_nudged(self, env):
        reward, obs, meta, done = env._process_one("no tool call here", {})
        assert meta["invalid_action"] is True
        assert not done and env.exec_calls == []
        assert "tool call" in obs["content"]

    def test_last_turn_grades_after_exec(self, env):
        # On the final allowed turn a run_shell command execs then grades.
        reward, obs, meta, done = env._process_one(
            _call("run_shell", {"command": "ls"}), {"_turn": 0, "max_turns": 1}
        )
        assert done and reward == 1.0 and env.scored is True
        assert env.exec_calls == ["ls"]
        assert meta["invalid_action"] is False


# ── legacy path unchanged when the flag is off ──────────────────────


class TestLegacyPathByteIdentical:
    LEGACY_BASH = "let me run it\n```bash\nls -la\n```\n"
    LEGACY_DONE = "all set\nTASK_COMPLETE\n"

    def test_parse_action_unchanged_command(self):
        assert _parse_action(self.LEGACY_BASH) == ("ls -la", False)

    def test_parse_action_unchanged_done(self):
        assert _parse_action(self.LEGACY_DONE) == ("", True)

    def test_flag_off_uses_legacy_parser(self, monkeypatch):
        # With structured_tools off, a Hermes <tool_call> is NOT parsed as an
        # action — the fenced parser sees no bash block and stamps invalid.
        exec_calls: list[str] = []
        monkeypatch.setattr(mtse, "start_session", lambda *a, **k: "sid")
        monkeypatch.setattr(mtse, "delete_session", lambda *a, **k: None)
        monkeypatch.setattr(
            mtse,
            "session_exec",
            lambda url, sid, command, **k: exec_calls.append(command)
            or {"stdout": "", "stderr": "", "exit_code": 0, "timed_out": False},
        )
        e = _StubSessionEnv({"sandbox_urls": ["http://x"], "max_turns": 30})
        assert e.structured_tools is False
        _, _, meta, done = e._process_one(_call("run_shell", {"command": "ls"}), {})
        assert exec_calls == [] and not done
        assert meta["invalid_action"] is True

    def test_flag_off_legacy_bash_execs(self, env, monkeypatch):
        # And a legacy fenced block still execs under the off flag.
        exec_calls: list[str] = []
        monkeypatch.setattr(mtse, "start_session", lambda *a, **k: "sid")
        monkeypatch.setattr(mtse, "delete_session", lambda *a, **k: None)
        monkeypatch.setattr(
            mtse,
            "session_exec",
            lambda url, sid, command, **k: exec_calls.append(command)
            or {"stdout": "OUT", "stderr": "", "exit_code": 0, "timed_out": False},
        )
        e = _StubSessionEnv({"sandbox_urls": ["http://x"], "max_turns": 30})
        e._process_one(self.LEGACY_BASH, {})
        assert exec_calls == ["ls -la"]


# ── CodeEnvironment patch extraction ────────────────────────────────


class TestCodePatchExtraction:
    PATCH = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"

    def test_submit_patch_valid(self):
        patch, invalid = _extract_patch_structured(
            _call("submit_patch", {"patch": self.PATCH})
        )
        assert patch == self.PATCH and invalid is False

    def test_submit_patch_empty_invalid(self):
        patch, invalid = _extract_patch_structured(
            _call("submit_patch", {"patch": "   "})
        )
        assert invalid is True

    def test_no_call_invalid(self):
        patch, invalid = _extract_patch_structured("here is my reasoning")
        assert patch == "" and invalid is True

    def test_unknown_tool_invalid(self):
        patch, invalid = _extract_patch_structured(_call("run_shell", {"command": "x"}))
        assert invalid is True

    def test_malformed_invalid(self):
        text = HERMES_TOOL_CALL_START + "\n{nope}\n</tool_call>"
        patch, invalid = _extract_patch_structured(text)
        assert invalid is True

    def test_missing_patch_arg_invalid(self):
        patch, invalid = _extract_patch_structured(_call("submit_patch", {}))
        assert invalid is True

    def test_multiple_calls_invalid(self):
        text = _call("submit_patch", {"patch": self.PATCH}) + _call(
            "submit_patch", {"patch": self.PATCH}
        )
        patch, invalid = _extract_patch_structured(text)
        assert invalid is True

    def test_legacy_extract_unchanged_diff_fence(self):
        text = "```diff\n" + self.PATCH + "```"
        assert _extract_patch(text) == self.PATCH.strip()

    def test_legacy_extract_unchanged_patch_tag(self):
        text = "<patch>" + self.PATCH + "</patch>"
        assert _extract_patch(text) == self.PATCH.strip()

    def test_legacy_extract_unchanged_raw_diff(self):
        assert _extract_patch(self.PATCH) == self.PATCH.strip()

    def test_legacy_ignores_structured_call(self):
        # A Hermes tool call is not a diff fence/tag/raw-diff → legacy yields "".
        assert _extract_patch(_call("submit_patch", {"patch": self.PATCH})) == ""
