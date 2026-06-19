"""Tests for the structured tool-use constrained-decoding plumbing (slice 7).

Pure-CPU coverage of fork 2's RL-safe constrained-decoding mechanism:

  * ``build_structural_tag`` produces the JSON structural-tag payload vLLM 0.21's
    xgrammar backend accepts (the ``structures``/``triggers`` legacy shape), pins
    the call object's ``name``/``arguments`` schema, and triggers on
    ``<tool_call>``.
  * ``_build_sampling_params`` is byte-identical to the unconstrained path when no
    ``structured_spec`` is supplied, and sets ``structured_outputs`` only when one
    is — exercised against the real (unbound) worker method bound to a stub so no
    Ray/GPU/engine is needed.
  * ``derive_structural_submask`` marks exactly the ``<tool_call>``/``</tool_call>``
    delimiter token spans (the CPU-derivable part of the loss-mask) and nothing
    else.

The live grammar backend (xgrammar) is not installed in the base dev env, so the
end-to-end "vLLM actually accepts and compiles the tag" assertion is guarded with
``pytest.importorskip`` and otherwise hardware-deferred.
"""

from __future__ import annotations

import json

import pytest

from dockyard_rl.models.generation.vllm.utils import (
    build_structural_tag,
    derive_structural_submask,
)
from dockyard_rl.tool_protocol.registry import CODE_TOOLS, SESSION_TOOLS, ToolRegistry


# ── build_structural_tag ────────────────────────────────────────────


def test_structural_tag_is_valid_legacy_shape():
    tag = build_structural_tag(CODE_TOOLS)
    payload = json.loads(tag)
    assert set(payload) == {"structures", "triggers"}
    assert payload["triggers"] == ["<tool_call>"]
    assert len(payload["structures"]) == 1
    struct = payload["structures"][0]
    assert set(struct) == {"begin", "schema", "end"}
    assert struct["begin"] == "<tool_call>\n"
    assert struct["end"] == "\n</tool_call>"


def test_structural_tag_pins_name_and_arguments_schema():
    tag = build_structural_tag(CODE_TOOLS)
    schema = json.loads(tag)["structures"][0]["schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["name", "arguments"]
    # name is pinned to the tool via const.
    assert schema["properties"]["name"]["const"] == "submit_patch"
    # arguments is exactly the tool's own parameter schema (string value left free).
    args = schema["properties"]["arguments"]
    assert args == CODE_TOOLS.parameters("submit_patch")
    assert args["properties"]["patch"]["type"] == "string"
    assert "const" not in args["properties"]["patch"]
    assert "enum" not in args["properties"]["patch"]


def test_structural_tag_subset_selection():
    tag = build_structural_tag(SESSION_TOOLS, ["run_shell", "task_complete"])
    payload = json.loads(tag)
    names = [s["schema"]["properties"]["name"]["const"] for s in payload["structures"]]
    assert names == ["run_shell", "task_complete"]


def test_structural_tag_all_tools_by_default():
    tag = build_structural_tag(SESSION_TOOLS)
    payload = json.loads(tag)
    names = {s["schema"]["properties"]["name"]["const"] for s in payload["structures"]}
    assert names == set(SESSION_TOOLS.names())


def test_structural_tag_rejects_empty_and_unknown():
    with pytest.raises(ValueError):
        build_structural_tag(CODE_TOOLS, [])
    with pytest.raises(ValueError):
        build_structural_tag(CODE_TOOLS, ["no_such_tool"])


def test_structural_tag_is_deterministic():
    # Stable serialization (sort_keys) so identical specs hash/compare equal.
    assert build_structural_tag(SESSION_TOOLS) == build_structural_tag(SESSION_TOOLS)


def test_structural_tag_handles_no_param_tool():
    # task_complete has empty parameters; the call object must still be valid.
    tag = build_structural_tag(SESSION_TOOLS, ["task_complete"])
    schema = json.loads(tag)["structures"][0]["schema"]
    assert schema["properties"]["arguments"] == SESSION_TOOLS.parameters("task_complete")


@pytest.mark.parametrize("registry", [CODE_TOOLS, SESSION_TOOLS])
def test_structural_tag_compiles_when_xgrammar_present(registry):
    # Hardware-deferred unless xgrammar is installed: confirm the real vLLM 0.21
    # validator accepts the payload shape we emit.
    pytest.importorskip("xgrammar")
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams
    from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

    tag = build_structural_tag(registry)
    # validate_xgrammar_grammar reads sampling_params.structured_outputs, so the
    # tag must be wrapped in a SamplingParams (passing the bare params object
    # would AttributeError on .structured_outputs).
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(structural_tag=tag)
    )
    validate_xgrammar_grammar(params)  # raises ValueError on a bad shape


# ── _build_sampling_params byte-identity ────────────────────────────


class _StubWorker:
    """Minimal carrier for the real, unbound ``_build_sampling_params`` method.

    Holds exactly the attributes the method reads (``cfg``, ``SamplingParams``,
    ``StructuredOutputsParams``) so the param-construction logic is exercised
    without constructing a vLLM engine.
    """

    def __init__(self):
        from vllm.sampling_params import SamplingParams, StructuredOutputsParams

        self.SamplingParams = SamplingParams
        self.StructuredOutputsParams = StructuredOutputsParams
        self.cfg = {
            "top_k": 20,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 256,
            "stop_token_ids": [151645],
        }


def _build(worker, **kwargs):
    from dockyard_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker

    return BaseVllmGenerationWorker._build_sampling_params(worker, **kwargs)


def _require_vllm():
    pytest.importorskip("vllm")


def test_sampling_params_byte_identical_when_off():
    _require_vllm()
    worker = _StubWorker()
    # No structured_spec at all, and explicit None: both must equal the baseline.
    baseline = _build(worker, greedy=False, stop_strings=None)
    explicit_none = _build(worker, greedy=False, stop_strings=None, structured_spec=None)

    assert baseline.structured_outputs is None
    assert explicit_none.structured_outputs is None
    # vLLM SamplingParams compares structurally (it is a dataclass/msgspec struct).
    assert repr(baseline) == repr(explicit_none)


def test_sampling_params_empty_tag_is_noop():
    _require_vllm()
    worker = _StubWorker()
    baseline = _build(worker, greedy=False, stop_strings=None)
    # A spec whose structural_tag is empty/missing must not set structured_outputs.
    for spec in ({"structural_tag": "", "constrained_tools": []}, {"constrained_tools": []}):
        params = _build(worker, greedy=False, stop_strings=None, structured_spec=spec)
        assert params.structured_outputs is None
        assert repr(params) == repr(baseline)


def test_sampling_params_sets_structured_outputs_when_present():
    _require_vllm()
    worker = _StubWorker()
    tag = build_structural_tag(CODE_TOOLS)
    spec = {"structural_tag": tag, "constrained_tools": ["submit_patch"]}
    params = _build(worker, greedy=False, stop_strings=None, structured_spec=spec)
    assert params.structured_outputs is not None
    assert params.structured_outputs.structural_tag == tag
    # Only the structural_tag constraint is set; all other constraints are None.
    assert params.structured_outputs.all_non_structural_tag_constraints_none()


def test_sampling_params_constrained_preserves_other_fields():
    _require_vllm()
    worker = _StubWorker()
    baseline = _build(worker, greedy=False, stop_strings=["X"], max_new_tokens=42)
    spec = {"structural_tag": build_structural_tag(CODE_TOOLS), "constrained_tools": ["submit_patch"]}
    constrained = _build(
        worker, greedy=False, stop_strings=["X"], max_new_tokens=42, structured_spec=spec
    )
    # Every sampling field except structured_outputs is unchanged by constraining.
    assert constrained.temperature == baseline.temperature
    assert constrained.top_p == baseline.top_p
    assert constrained.top_k == baseline.top_k
    assert constrained.max_tokens == baseline.max_tokens
    assert constrained.stop == baseline.stop
    assert constrained.stop_token_ids == baseline.stop_token_ids
    assert constrained.include_stop_str_in_output == baseline.include_stop_str_in_output


# ── ignore_eos passthrough ──────────────────────────────────────────


def test_sampling_params_ignore_eos_default_false():
    _require_vllm()
    worker = _StubWorker()  # cfg carries no ignore_eos key
    params = _build(worker, greedy=False, stop_strings=None)
    assert params.ignore_eos is False


def test_sampling_params_ignore_eos_respected():
    _require_vllm()
    worker = _StubWorker()
    worker.cfg["ignore_eos"] = True
    params = _build(worker, greedy=False, stop_strings=None)
    assert params.ignore_eos is True


# ── derive_structural_submask ───────────────────────────────────────


def test_submask_marks_only_delimiter_spans():
    # Synthetic token ids: <tool_call> = [10, 11], </tool_call> = [20, 21].
    begin = [10, 11]
    end = [20, 21]
    # [pre][<tool_call>][json...][</tool_call>][post]
    gen = [1, 2, 10, 11, 99, 98, 97, 20, 21, 3]
    mask = derive_structural_submask(gen, begin, end)
    assert mask == [0, 0, 1, 1, 0, 0, 0, 1, 1, 0]


def test_submask_multiple_calls():
    begin = [10]
    end = [20]
    gen = [10, 5, 20, 7, 10, 6, 20]
    mask = derive_structural_submask(gen, begin, end)
    assert mask == [1, 0, 1, 0, 1, 0, 1]


def test_submask_no_tool_call_is_all_zero():
    mask = derive_structural_submask([1, 2, 3, 4], [10, 11], [20, 21])
    assert mask == [0, 0, 0, 0]


def test_submask_length_matches_input():
    gen = list(range(50))
    mask = derive_structural_submask(gen, [10], [20])
    assert len(mask) == len(gen)


def test_submask_empty_delimiters_are_ignored():
    gen = [1, 2, 3]
    assert derive_structural_submask(gen, [], []) == [0, 0, 0]


def test_submask_overlapping_delimiter_value_collision():
    # When a delimiter token id also appears as free content it is still only
    # marked where the full delimiter subsequence matches.
    begin = [10, 11]
    end = [20]
    gen = [10, 11, 10, 99, 20]  # second lone 10 is content, not a delimiter
    mask = derive_structural_submask(gen, begin, end)
    assert mask == [1, 1, 0, 0, 1]
