"""Unit tests for structured tool-use prompt construction (S6).

Pure CPU coverage of the prompt-construction wiring: when the
``grpo.structured_tool_use`` gate is enabled the SWE / session processors
advertise their environment's tool registry via the chat template's ``tools=``
argument; when it is disabled (the default) the chat-template call is exactly the
legacy one (no ``tools`` passed).

The processors delegate templating to ``data.llm_message_utils.
get_formatted_message_log``, which forwards ``tools=`` to
``tokenizer.apply_chat_template``. These tests patch that delegate with a
recorder so the wiring is asserted without a real Qwen tokenizer (none is cached
in this environment): which registry each env selects, that the gate is honoured,
and that flag-off reproduces the legacy call args byte-for-byte.
"""

import json
from typing import Any

import pytest
import torch

import dockyard_rl.data.processors as processors
from dockyard_rl.data.interfaces import TaskDataSpec
from dockyard_rl.data.processors import (
    _resolve_structured_tools,
    gdpval_agentic_data_processor,
    program_bench_data_processor,
    swe_bench_data_processor,
    swe_bench_pro_data_processor,
    terminal_bench_data_processor,
)
from dockyard_rl.tool_protocol.protocol import STRUCTURED_TOOL_USE_KEY
from dockyard_rl.tool_protocol.registry import CODE_TOOLS, SESSION_TOOLS

# Stand-in tokenizer: on these wiring paths the processors never call it
# (templating is monkeypatched), so an opaque object suffices; typed Any to
# satisfy the strict ``tokenizer`` parameter.
_FAKE_TOK: Any = object()


def _spec(structured_tool_use=None) -> TaskDataSpec:
    """A minimal TaskDataSpec with no prompt/system-prompt files.

    The structured-tool-use gate is read off the spec by ``getattr`` (the
    dataset-loading layer attaches the resolved ``grpo.structured_tool_use``
    block there), so the tests attach it directly as an attribute.
    """
    spec = TaskDataSpec(task_name="t")
    if structured_tool_use is not None:
        setattr(spec, STRUCTURED_TOOL_USE_KEY, structured_tool_use)
    return spec


class _RecordingGetFormattedMessageLog:
    """Stand-in for ``get_formatted_message_log`` that records its kwargs.

    Returns a single-message log with a non-empty ``token_ids`` tensor so the
    processors' length/loss-multiplier logic runs unchanged.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, message_log, tokenizer, task_data_spec, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "role": "user",
                "content": "x",
                "token_ids": torch.tensor([1, 2, 3], dtype=torch.int64),
            }
        ]


@pytest.fixture
def recorder(monkeypatch):
    rec = _RecordingGetFormattedMessageLog()
    monkeypatch.setattr(processors, "get_formatted_message_log", rec)
    return rec


# Complete datum rows: the processors read these keys to build extra_env_info.
SWE_ROW = {
    "messages": [{"role": "user", "content": "fix it"}],
    "instance_id": "i", "repo": "r", "base_commit": "c",
    "environment_setup_commit": "e", "fail_to_pass": "[]", "pass_to_pass": "[]",
    "test_patch": "", "gold_patch": "",
}

SWE_PRO_ROW = {
    "messages": [{"role": "user", "content": "fix it"}],
    "instance_id": "i", "repo": "r", "base_commit": "c", "repo_language": "py",
    "image": "img", "before_repo_set_cmd": "", "selected_test_files": "[]",
    "fail_to_pass": "[]", "pass_to_pass": "[]", "test_patch": "", "gold_patch": "",
    "run_script": "", "parser_script": "",
}

TERMINAL_ROW = {
    "messages": [{"role": "user", "content": "do it"}],
    "task_id": "t", "mode": "m", "image": "img", "dockerfile": "FROM x",
    "build_context_json": "{}", "tests_json": "{}", "container_env_json": "{}",
    "allow_internet": False, "harness_mount": "/mnt", "test_command": "pytest",
    "result_files": "[]", "agent_timeout_sec": 1, "verifier_timeout_sec": 1,
    "build_timeout_sec": 1, "exec_timeout_sec": 1, "max_turns": 5,
}

PROGRAM_ROW = {
    "messages": [{"role": "user", "content": "do it"}],
    "task_id": "t", "language": "py", "image": "img", "branches_json": "{}",
    "max_turns": 5, "exec_timeout_sec": 1, "grading_timeout_sec": 1,
}

GDPVAL_AGENTIC_ROW = {
    "messages": [{"role": "user", "content": "do it"}],
    "prompt": "p", "rubric_json": "{}", "occupation": "o", "sector": "s",
    "task_id": "t", "image": "img", "deliverable_dir": "/d", "reference_dir": "/r",
    "max_turns": 5, "exec_timeout_sec": 1, "reference_files_json": "{}",
}

# (processor, datum_row, expected_registry)
_CODE_CASES = [
    (swe_bench_data_processor, SWE_ROW, CODE_TOOLS),
    (swe_bench_pro_data_processor, SWE_PRO_ROW, CODE_TOOLS),
]
_SESSION_CASES = [
    (terminal_bench_data_processor, TERMINAL_ROW, SESSION_TOOLS),
    (program_bench_data_processor, PROGRAM_ROW, SESSION_TOOLS),
    (gdpval_agentic_data_processor, GDPVAL_AGENTIC_ROW, SESSION_TOOLS),
]
_ALL_CASES = _CODE_CASES + _SESSION_CASES


class TestResolveStructuredTools:
    def test_disabled_returns_none_when_attribute_absent(self):
        assert _resolve_structured_tools(_spec(), CODE_TOOLS) is None

    def test_disabled_returns_none_when_block_disabled(self):
        spec = _spec({"enabled": False})
        assert _resolve_structured_tools(spec, CODE_TOOLS) is None

    def test_enabled_returns_registry_specs(self):
        spec = _spec({"enabled": True})
        tools = _resolve_structured_tools(spec, CODE_TOOLS)
        assert tools == CODE_TOOLS.chat_template_tools()

    def test_enabled_session_registry(self):
        spec = _spec({"enabled": True})
        tools = _resolve_structured_tools(spec, SESSION_TOOLS)
        assert tools == SESSION_TOOLS.chat_template_tools()
        assert tools is not None
        names = [t["function"]["name"] for t in tools]
        assert names == ["run_shell", "read_file", "write_file", "task_complete"]

    def test_invalid_thinking_style_raises(self):
        spec = _spec({"enabled": True, "thinking_style": "bogus"})
        with pytest.raises(ValueError):
            _resolve_structured_tools(spec, CODE_TOOLS)


class TestGateOffByteIdentical:
    """Flag-off must reproduce the legacy call: no ``tools`` kwarg at all."""

    @pytest.mark.parametrize("proc,row,_reg", _ALL_CASES)
    def test_no_gate_attribute_omits_tools(self, recorder, proc, row, _reg):
        proc(row, _spec(), tokenizer=_FAKE_TOK, max_seq_length=4096, idx=0)
        assert len(recorder.calls) == 1
        # Legacy path: tools is None (delegate omits the kwarg from the template).
        assert recorder.calls[0].get("tools") is None

    @pytest.mark.parametrize("proc,row,_reg", _ALL_CASES)
    def test_disabled_block_omits_tools(self, recorder, proc, row, _reg):
        proc(row, _spec({"enabled": False}), tokenizer=_FAKE_TOK,
             max_seq_length=4096, idx=0)
        assert recorder.calls[0].get("tools") is None

    @pytest.mark.parametrize("proc,row,_reg", _ALL_CASES)
    def test_off_matches_legacy_kwargs(self, recorder, proc, row, _reg):
        # Capture the legacy (gate absent) kwargs, then the disabled-block kwargs;
        # both must equal the pre-slice call signature exactly.
        proc(row, _spec(), tokenizer=_FAKE_TOK, max_seq_length=4096, idx=0)
        legacy = dict(recorder.calls[-1])
        legacy.pop("tools", None)  # legacy never carried a real tools list
        assert legacy == {"add_eos_token": False, "add_generation_prompt": True}


class TestGateOnSelectsRegistry:
    @pytest.mark.parametrize("proc,row,reg", _ALL_CASES)
    def test_enabled_passes_registry_specs(self, recorder, proc, row, reg):
        proc(row, _spec({"enabled": True}), tokenizer=_FAKE_TOK,
             max_seq_length=4096, idx=0)
        assert len(recorder.calls) == 1
        passed = recorder.calls[0]["tools"]
        assert passed == reg.chat_template_tools()

    def test_code_envs_select_code_tools_not_session(self, recorder):
        for proc, row, _reg in _CODE_CASES:
            recorder.calls.clear()
            proc(row, _spec({"enabled": True}), tokenizer=_FAKE_TOK,
                 max_seq_length=4096, idx=0)
            passed = recorder.calls[0]["tools"]
            assert passed == CODE_TOOLS.chat_template_tools()
            assert passed != SESSION_TOOLS.chat_template_tools()
            names = [t["function"]["name"] for t in passed]
            assert names == ["submit_patch"]

    def test_session_envs_select_session_tools_not_code(self, recorder):
        for proc, row, _reg in _SESSION_CASES:
            recorder.calls.clear()
            proc(row, _spec({"enabled": True}), tokenizer=_FAKE_TOK,
                 max_seq_length=4096, idx=0)
            passed = recorder.calls[0]["tools"]
            assert passed == SESSION_TOOLS.chat_template_tools()
            assert passed != CODE_TOOLS.chat_template_tools()


class TestThinkingChatTemplateKwargs:
    """Thinking config (``thinking_style`` / ``enable_thinking``) is honoured.

    Per-call ``chat_template_kwargs`` (e.g. ``enable_thinking``) are applied at
    tokenizer-init time in ``algorithms.utils.get_tokenizer`` (partial-applied
    onto ``apply_chat_template``), so they reach every processor template call
    without per-processor forwarding. The processor-level gate only governs
    whether the tool catalogue is advertised; a valid thinking_style must not
    disturb tool selection.
    """

    @pytest.mark.parametrize("proc,row,reg", _ALL_CASES)
    def test_thinking_style_does_not_change_tools(self, recorder, proc, row, reg):
        proc(row, _spec({"enabled": True, "thinking_style": "closing_only"}),
             tokenizer=_FAKE_TOK, max_seq_length=4096, idx=0)
        assert recorder.calls[0]["tools"] == reg.chat_template_tools()


def test_extra_env_info_unchanged_when_gate_on(recorder):
    """Enabling the gate only adds tool advertisement; env scoring inputs are
    untouched (regression guard on the held-out metadata)."""
    out = swe_bench_data_processor(
        SWE_ROW, _spec({"enabled": True}), tokenizer=_FAKE_TOK,
        max_seq_length=4096, idx=0,
    )
    assert out is not None
    env_info = out["extra_env_info"]
    assert env_info is not None
    assert env_info["instance_id"] == "i"
    assert env_info["test_patch"] == ""
    # The patch-scoring metadata never leaks into the advertised tools.
    assert "test_patch" not in json.dumps(recorder.calls[0]["tools"])
