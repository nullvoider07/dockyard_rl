"""Unit tests for the structured tool-use protocol primitives (S1).

Pure CPU coverage of the Hermes extractor and the dependency-free JSON-schema
validator, plus a best-effort parity check against vLLM's real
``Hermes2ProToolParser`` (skipped when vLLM or a usable tokenizer is absent).
"""

import json

import pytest

from dockyard_rl.tool_protocol import (
    CODE_TOOLS,
    HERMES_TOOL_CALL_START,
    SESSION_TOOLS,
    SchemaError,
    ToolRegistry,
    assess_tool_calls,
    make_tool,
    parse_hermes_tool_calls,
    resolve_structured_tool_use,
    validate_against_schema,
    validate_arguments,
)


def _call(name: str, args: dict) -> str:
    return (
        HERMES_TOOL_CALL_START
        + "\n"
        + json.dumps({"name": name, "arguments": args})
        + "\n</tool_call>"
    )


# Hermes extractor
class TestParseHermesToolCalls:
    def test_no_tool_call(self):
        ext = parse_hermes_tool_calls("just thinking, no action")
        assert not ext.tools_called
        assert ext.tool_calls == []
        assert ext.content == "just thinking, no action"
        assert ext.error is None
        assert not ext.attempted

    def test_single_call(self):
        ext = parse_hermes_tool_calls(_call("run_shell", {"command": "ls -la"}))
        assert ext.tools_called
        assert ext.error is None
        assert ext.attempted
        assert len(ext.tool_calls) == 1
        call = ext.tool_calls[0]
        assert call.name == "run_shell"
        assert call.arguments == {"command": "ls -la"}
        assert json.loads(call.arguments_json) == {"command": "ls -la"}

    def test_leading_content_preserved(self):
        text = "let me look around.\n" + _call("run_shell", {"command": "pwd"})
        ext = parse_hermes_tool_calls(text)
        assert ext.tools_called
        assert ext.content == "let me look around.\n"

    def test_leading_content_none_when_empty(self):
        ext = parse_hermes_tool_calls(_call("task_complete", {}))
        assert ext.tools_called
        assert ext.content is None

    def test_multiple_calls(self):
        text = _call("read_file", {"path": "a.py"}) + _call("read_file", {"path": "b.py"})
        ext = parse_hermes_tool_calls(text)
        assert ext.tools_called
        assert [c.arguments["path"] for c in ext.tool_calls] == ["a.py", "b.py"]

    def test_malformed_json_fails_all(self):
        text = HERMES_TOOL_CALL_START + "\n{not json}\n</tool_call>"
        ext = parse_hermes_tool_calls(text)
        # vLLM parity: any decode error -> tools_called False, content = full text.
        assert not ext.tools_called
        assert ext.tool_calls == []
        assert ext.content == text
        assert ext.error is not None
        assert ext.attempted  # a call was attempted, just malformed

    def test_missing_name_key_fails(self):
        text = HERMES_TOOL_CALL_START + "\n" + json.dumps({"arguments": {}}) + "\n</tool_call>"
        ext = parse_hermes_tool_calls(text)
        assert not ext.tools_called
        assert ext.error is not None
        assert ext.attempted

    def test_missing_arguments_key_fails(self):
        text = HERMES_TOOL_CALL_START + "\n" + json.dumps({"name": "x"}) + "\n</tool_call>"
        ext = parse_hermes_tool_calls(text)
        assert not ext.tools_called
        assert ext.error is not None

    def test_one_bad_call_invalidates_batch(self):
        # Mirrors vLLM's all-or-nothing behaviour: a valid call followed by an
        # unparseable one yields no tool calls at all.
        text = _call("run_shell", {"command": "ls"}) + HERMES_TOOL_CALL_START + "\n{bad\n</tool_call>"
        ext = parse_hermes_tool_calls(text)
        assert not ext.tools_called
        assert ext.error is not None

    def test_unterminated_trailing_region(self):
        # Truncated generation: <tool_call> with no close. The second regex
        # alternative captures to end-of-string (vLLM behaviour).
        text = HERMES_TOOL_CALL_START + "\n" + json.dumps({"name": "run_shell", "arguments": {"command": "x"}})
        ext = parse_hermes_tool_calls(text)
        assert ext.tools_called
        assert ext.tool_calls[0].name == "run_shell"

    def test_thinking_then_tool_call(self):
        # Qwen3 interleaved thinking: <think>…</think> precedes the call.
        text = "<think>I should list files</think>\n" + _call("run_shell", {"command": "ls"})
        ext = parse_hermes_tool_calls(text)
        assert ext.tools_called
        assert ext.tool_calls[0].arguments == {"command": "ls"}
        assert ext.content is not None and "<think>" in ext.content

    def test_tool_call_inside_unclosed_think(self):
        # Qwen3.5 implicit-close: <tool_call> emitted before </think>. Extraction
        # must still find the call (the malformed-thinking detector handles the
        # implicit close separately).
        text = "<think>let me run it\n" + _call("run_shell", {"command": "ls"})
        ext = parse_hermes_tool_calls(text)
        assert ext.tools_called
        assert ext.tool_calls[0].name == "run_shell"


# schema validator
class TestValidateArguments:
    SCHEMA = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": ["command"],
    }

    def test_valid(self):
        assert validate_arguments({"command": "ls", "timeout": 30}, self.SCHEMA) == []

    def test_missing_required(self):
        errs = validate_arguments({"timeout": 30}, self.SCHEMA)
        assert len(errs) == 1
        assert "command" in errs[0]

    def test_wrong_type(self):
        errs = validate_arguments({"command": 123}, self.SCHEMA)
        assert any("expected string" in e for e in errs)

    def test_arguments_not_object(self):
        errs = validate_arguments(["command"], self.SCHEMA)
        assert errs and "expected object" in errs[0]

    def test_bool_is_not_integer(self):
        errs = validate_arguments({"command": "ls", "timeout": True}, self.SCHEMA)
        assert any("expected integer" in e for e in errs)

    def test_integer_is_valid_number(self):
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        assert validate_arguments({"x": 5}, schema) == []

    def test_enum(self):
        schema = {"type": "object", "properties": {"mode": {"enum": ["r", "w"]}}}
        assert validate_arguments({"mode": "r"}, schema) == []
        assert validate_arguments({"mode": "x"}, schema)

    def test_nested_object(self):
        schema = {
            "type": "object",
            "properties": {
                "opts": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                }
            },
        }
        assert validate_arguments({"opts": {"n": 1}}, schema) == []
        assert validate_arguments({"opts": {}}, schema)  # missing nested required

    def test_array_items(self):
        schema = {
            "type": "object",
            "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
        }
        assert validate_arguments({"paths": ["a", "b"]}, schema) == []
        assert validate_arguments({"paths": ["a", 2]}, schema)

    def test_unsupported_type_raises(self):
        with pytest.raises(SchemaError):
            validate_against_schema("x", {"type": "bogus"})


# tool registry + assess bridge (S3)
class TestToolRegistry:
    def test_base_session_tools(self):
        assert set(SESSION_TOOLS.names()) == {
            "run_shell", "read_file", "write_file", "task_complete"
        }
        assert "run_shell" in SESSION_TOOLS
        assert "submit_patch" not in SESSION_TOOLS

    def test_code_tools(self):
        assert CODE_TOOLS.names() == ["submit_patch"]
        params = CODE_TOOLS.parameters("submit_patch")
        assert params is not None and params["required"] == ["patch"]

    def test_chat_template_tools_shape(self):
        specs = SESSION_TOOLS.chat_template_tools()
        assert all(s["type"] == "function" for s in specs)
        assert all("parameters" in s["function"] for s in specs)

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError):
            ToolRegistry(
                tools=(
                    make_tool("x", "d", {"type": "object", "properties": {}}),
                    make_tool("x", "d", {"type": "object", "properties": {}}),
                )
            )


class TestAssessToolCalls:
    def test_no_call(self):
        a = assess_tool_calls("just thinking", SESSION_TOOLS)
        assert not a.attempted
        assert not a.malformed
        assert not a.has_valid_call

    def test_valid_call(self):
        a = assess_tool_calls(_call("run_shell", {"command": "ls"}), SESSION_TOOLS)
        assert a.attempted and a.has_valid_call and not a.malformed
        assert a.valid_calls[0].name == "run_shell"

    def test_unknown_tool(self):
        a = assess_tool_calls(_call("delete_everything", {}), SESSION_TOOLS)
        assert a.attempted and a.malformed and not a.has_valid_call
        assert a.reason is not None and "unknown tool" in a.reason

    def test_schema_invalid_args(self):
        a = assess_tool_calls(_call("run_shell", {"command": 123}), SESSION_TOOLS)
        assert a.malformed and not a.has_valid_call
        assert a.reason is not None and "run_shell" in a.reason

    def test_missing_required_arg(self):
        a = assess_tool_calls(_call("write_file", {"path": "a"}), SESSION_TOOLS)
        assert a.malformed
        assert a.reason is not None and "content" in a.reason

    def test_malformed_json(self):
        text = HERMES_TOOL_CALL_START + "\n{bad}\n</tool_call>"
        a = assess_tool_calls(text, SESSION_TOOLS)
        assert a.attempted and a.malformed

    def test_task_complete_no_args(self):
        a = assess_tool_calls(_call("task_complete", {}), SESSION_TOOLS)
        assert a.has_valid_call and not a.malformed


# protocol config resolver
class TestResolveStructuredToolUse:
    def test_none_and_disabled(self):
        assert resolve_structured_tool_use(None) is None
        assert resolve_structured_tool_use({"enabled": False}) is None

    def test_enabled_defaults(self):
        cfg = resolve_structured_tool_use({"enabled": True})
        assert cfg is not None

    def test_bad_thinking_style_raises(self):
        with pytest.raises(ValueError):
            resolve_structured_tool_use({"enabled": True, "thinking_style": "bogus"})

    def test_non_vllm_backend_raises(self):
        with pytest.raises(ValueError):
            resolve_structured_tool_use({"enabled": True, "backend": "sglang"})


# best-effort parity vs vLLM's real Hermes parser
class TestVllmParity:
    """Assert byte-parity with vLLM's Hermes2ProToolParser where available.

    Skipped cleanly when vLLM, its OpenAI protocol chain, or a usable tokenizer
    is not importable in the environment.
    """

    @staticmethod
    def _vllm_parser():
        hermes = pytest.importorskip("vllm.tool_parsers.hermes_tool_parser")
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415

            tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        except Exception:  # noqa: BLE001 — no network/cache: skip parity
            pytest.skip("no tokenizer available for vLLM Hermes parser construction")
        return hermes.Hermes2ProToolParser(tok)

    @pytest.mark.parametrize(
        "text",
        [
            "no tool call here",
            _call("run_shell", {"command": "ls -la"}),
            "thinking\n" + _call("task_complete", {}),
            _call("read_file", {"path": "a"}) + _call("read_file", {"path": "b"}),
            HERMES_TOOL_CALL_START + "\n{bad json}\n</tool_call>",
        ],
    )
    def test_parity(self, text):
        parser = self._vllm_parser()
        # Hermes' non-streaming extract_tool_calls ignores the request arg.
        vllm_out = parser.extract_tool_calls(text, None)
        ours = parse_hermes_tool_calls(text)

        assert ours.tools_called == vllm_out.tools_called
        assert ours.content == vllm_out.content
        if vllm_out.tools_called:
            assert [c.name for c in ours.tool_calls] == [
                c.function.name for c in vllm_out.tool_calls
            ]
            assert [c.arguments_json for c in ours.tool_calls] == [
                c.function.arguments for c in vllm_out.tool_calls
            ]
