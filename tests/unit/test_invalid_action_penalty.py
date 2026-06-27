"""Unit tests for the native invalid-action reward penalty (#2656).

Pure CPU coverage: generic detectors, env-flag composition, per-turn flag
hygiene, and the per-sample reward penalty arithmetic on BatchedDataDict.
"""

from typing import cast

import pytest
import torch

from dockyard_rl.algorithms.reward_functions import apply_invalid_action_penalty
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.rewards.invalid_action import (
    ENV_INVALID_ACTION_KEY,
    ENV_MALFORMED_THINKING_KEY,
    LOCUS_ADVANTAGE,
    LOCUS_REWARD,
    VIOLATION_ENV_INVALID_ACTION,
    VIOLATION_MALFORMED_THINKING,
    VIOLATION_SCHEMA_ARGS,
    VIOLATION_SCHEMA_UNKNOWN_TOOL,
    VIOLATION_UNEXECUTED_PATTERN,
    InvalidActionPenaltyConfig,
    assess_assistant_turn,
    detect_invalid_tool_call,
    detect_malformed_thinking,
    penalty_scale_at_step,
    pop_env_flags,
)
from dockyard_rl.tool_protocol.registry import ToolRegistry, make_tool


# generic detectors
class TestDetectInvalidToolCall:
    def test_empty_patterns_never_flag(self):
        assert not detect_invalid_tool_call("<tool_call>run</tool_call>", [])

    def test_pattern_match_flags(self):
        assert detect_invalid_tool_call(
            "I will call <tool_call>ls</tool_call>", ["<tool_call>"]
        )

    def test_no_match(self):
        assert not detect_invalid_tool_call("plain answer", ["<tool_call>"])

    def test_empty_pattern_string_ignored(self):
        assert not detect_invalid_tool_call("anything", [""])


class TestDetectMalformedThinking:
    def test_no_tags_well_formed(self):
        assert not detect_malformed_thinking("just an answer")

    def test_single_balanced_pair_well_formed(self):
        assert not detect_malformed_thinking("<think>reasoning</think>answer")

    def test_unclosed_open_tag_malformed(self):
        assert detect_malformed_thinking("<think>reasoning without close")

    def test_dangling_close_tag_malformed(self):
        assert detect_malformed_thinking("leaked</think>answer")

    def test_repeated_pairs_malformed(self):
        assert detect_malformed_thinking(
            "<think>a</think>mid<think>b</think>end"
        )

    def test_custom_tags(self):
        assert detect_malformed_thinking(
            "[REASON]x[/REASON][REASON]y[/REASON]", ["[REASON]", "[/REASON]"]
        )
        assert not detect_malformed_thinking(
            "[REASON]x[/REASON]", ["[REASON]", "[/REASON]"]
        )

    def test_bad_tag_config_raises(self):
        with pytest.raises(ValueError):
            detect_malformed_thinking("text", ["<think>"])

    # closing_only style (Qwen3.5: <think> is prompt-supplied, only </think>
    # appears in output).
    def test_closing_only_lone_close_well_formed(self):
        assert not detect_malformed_thinking(
            "reasoning</think>answer", style="closing_only"
        )

    def test_closing_only_no_tags_well_formed(self):
        # Implicit close before a tool call: zero </think> is fine.
        assert not detect_malformed_thinking("answer", style="closing_only")

    def test_closing_only_regenerated_open_malformed(self):
        assert detect_malformed_thinking(
            "<think>x</think>answer", style="closing_only"
        )

    def test_closing_only_double_close_malformed(self):
        assert detect_malformed_thinking(
            "a</think>b</think>c", style="closing_only"
        )

    def test_bad_style_raises(self):
        with pytest.raises(ValueError):
            detect_malformed_thinking("text", style="bogus")


# tier composition
_CFG: InvalidActionPenaltyConfig = {
    "enabled": True,
    "invalid_action_penalty": 0.5,
    "malformed_thinking_penalty": 0.25,
    "use_environment_flags": True,
    "invalid_tool_call_patterns": ["<tool_call>"],
    "thinking_tags": ["<think>", "</think>"],
}


class TestAssessAssistantTurn:
    def test_env_flag_authoritative(self):
        v = assess_assistant_turn(
            "clean text", _CFG, {ENV_INVALID_ACTION_KEY: True}
        )
        assert v.invalid_action and not v.malformed_thinking

    def test_env_malformed_flag(self):
        v = assess_assistant_turn(
            "clean text", _CFG, {ENV_MALFORMED_THINKING_KEY: True}
        )
        assert v.malformed_thinking and not v.invalid_action

    def test_generic_fallback_when_env_silent(self):
        v = assess_assistant_turn("try <tool_call>x</tool_call>", _CFG, {})
        assert v.invalid_action

    def test_generic_thinking_fallback(self):
        v = assess_assistant_turn("<think>oops no close", _CFG, None)
        assert v.malformed_thinking

    def test_env_flags_ignored_when_disabled(self):
        cfg = cast(
            InvalidActionPenaltyConfig, dict(_CFG, use_environment_flags=False)
        )
        v = assess_assistant_turn("clean", cfg, {ENV_INVALID_ACTION_KEY: True})
        assert not v.invalid_action

    def test_clean_turn(self):
        v = assess_assistant_turn(
            "<think>plan</think>the answer", _CFG, {ENV_INVALID_ACTION_KEY: False}
        )
        assert not v.invalid_action and not v.malformed_thinking

    def test_pop_env_flags(self):
        meta = {ENV_INVALID_ACTION_KEY: True, ENV_MALFORMED_THINKING_KEY: False, "x": 1}
        pop_env_flags(meta)
        assert meta == {"x": 1}
        pop_env_flags(None)  # no-op


# penalty arithmetic
def _batch(rewards, invalid=None, malformed=None):
    d = {"total_reward": torch.tensor(rewards, dtype=torch.float32)}
    if invalid is not None:
        d["invalid_action_count"] = torch.tensor(invalid, dtype=torch.int32)
    if malformed is not None:
        d["malformed_thinking_count"] = torch.tensor(malformed, dtype=torch.int32)
    return BatchedDataDict(d)


class TestApplyInvalidActionPenalty:
    def test_none_cfg_noop(self):
        b = _batch([1.0, 0.0])
        before = b["total_reward"].clone()
        out = apply_invalid_action_penalty(b, None)
        assert torch.equal(out["total_reward"], before)
        assert "unshaped_total_reward" not in out

    def test_disabled_noop(self):
        b = _batch([1.0, 0.0])
        before = b["total_reward"].clone()
        out = apply_invalid_action_penalty(b, {"enabled": False})
        assert torch.equal(out["total_reward"], before)

    def test_missing_counts_raises(self):
        with pytest.raises(ValueError, match="invalid_action_count"):
            apply_invalid_action_penalty(_batch([1.0]), {"enabled": True})

    def test_per_sample_subtraction_scales_with_counts(self):
        b = _batch([1.0, 1.0, 1.0], invalid=[0, 1, 2], malformed=[1, 0, 2])
        cfg: InvalidActionPenaltyConfig = {
            "enabled": True,
            "invalid_action_penalty": 0.5,
            "malformed_thinking_penalty": 0.25,
        }
        out = apply_invalid_action_penalty(b, cfg)
        assert torch.allclose(
            out["total_reward"], torch.tensor([0.75, 0.5, -0.5])
        )

    def test_unshaped_reward_preserved(self):
        b = _batch([1.0, 0.0], invalid=[1, 0], malformed=[0, 0])
        cfg: InvalidActionPenaltyConfig = {"enabled": True, "invalid_action_penalty": 0.5}
        out = apply_invalid_action_penalty(b, cfg)
        assert torch.allclose(
            out["unshaped_total_reward"], torch.tensor([1.0, 0.0])
        )
        assert torch.allclose(out["total_reward"], torch.tensor([0.5, 0.0]))

    def test_existing_unshaped_reward_not_overwritten(self):
        b = _batch([0.8, 0.0], invalid=[1, 0], malformed=[0, 0])
        b["unshaped_total_reward"] = torch.tensor([1.0, 0.0])
        cfg: InvalidActionPenaltyConfig = {"enabled": True, "invalid_action_penalty": 0.5}
        out = apply_invalid_action_penalty(b, cfg)
        assert torch.allclose(
            out["unshaped_total_reward"], torch.tensor([1.0, 0.0])
        )

    def test_reward_floor_clamps(self):
        b = _batch([0.0, 1.0], invalid=[3, 0], malformed=[0, 0])
        cfg: InvalidActionPenaltyConfig = {
            "enabled": True,
            "invalid_action_penalty": 1.0,
            "reward_floor": -1.0,
        }
        out = apply_invalid_action_penalty(b, cfg)
        assert torch.allclose(out["total_reward"], torch.tensor([-1.0, 1.0]))

    def test_zero_penalties_leave_rewards_unchanged(self):
        b = _batch([1.0, 0.5], invalid=[2, 0], malformed=[1, 1])
        out = apply_invalid_action_penalty(b, {"enabled": True})
        assert torch.allclose(out["total_reward"], torch.tensor([1.0, 0.5]))

    def test_negative_penalty_rejected(self):
        b = _batch([1.0], invalid=[1], malformed=[0])
        with pytest.raises(AssertionError):
            apply_invalid_action_penalty(
                b, {"enabled": True, "invalid_action_penalty": -0.5}
            )


# environment opt-in stamping
class TestEnvironmentStamping:
    def test_session_env_parse_action_verdict(self):
        # The MultiTurnSessionEnv stamps invalid_action when the turn yields
        # neither a runnable command nor a completion signal.
        from dockyard_rl.environments.multi_turn_session_env import _parse_action

        for text, expected_invalid in [
            ("```bash\nls\n```", False),
            ("TASK_COMPLETE", False),
            ("I think I should look around first.", True),
            ("```bash\nmake test\n```\nTASK_COMPLETE", False),
        ]:
            command, done = _parse_action(text)
            assert (not command and not done) == expected_invalid, text

    def test_code_env_extract_patch_verdict(self):
        # CodeEnvironment stamps invalid_action when no diff is extractable.
        from dockyard_rl.environments.code_environment import _extract_patch

        diff = "```diff\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+y\n```"
        assert _extract_patch(diff).strip()
        assert not _extract_patch("no patch here, just words").strip()


# Typed / graded violations (N4)
class TestTypedViolations:
    def test_env_invalid_is_reward_locus(self):
        v = assess_assistant_turn("clean", _CFG, {ENV_INVALID_ACTION_KEY: True})
        assert v.invalid_action
        types = {x.type: x for x in v.violations}
        assert VIOLATION_ENV_INVALID_ACTION in types
        # Env task-semantic invalid-action defaults to the reward locus.
        assert types[VIOLATION_ENV_INVALID_ACTION].locus == LOCUS_REWARD

    def test_pattern_is_advantage_locus(self):
        v = assess_assistant_turn("try <tool_call>x</tool_call>", _CFG, {})
        types = {x.type: x for x in v.violations}
        assert VIOLATION_UNEXECUTED_PATTERN in types
        # Structural/format violations default to the advantage (token-local) locus.
        assert types[VIOLATION_UNEXECUTED_PATTERN].locus == LOCUS_ADVANTAGE

    def test_thinking_is_advantage_locus(self):
        v = assess_assistant_turn("<think>oops no close", _CFG, None)
        types = {x.type for x in v.violations}
        assert VIOLATION_MALFORMED_THINKING in types
        assert all(
            x.locus == LOCUS_ADVANTAGE
            for x in v.violations
            if x.type == VIOLATION_MALFORMED_THINKING
        )

    def test_severity_default_and_override(self):
        v = assess_assistant_turn("try <tool_call>x</tool_call>", _CFG, {})
        sev = {x.type: x.severity for x in v.violations}
        assert sev[VIOLATION_UNEXECUTED_PATTERN] == 1.0  # default
        cfg = cast(
            InvalidActionPenaltyConfig,
            dict(_CFG, severity_weights={VIOLATION_UNEXECUTED_PATTERN: 2.5}),
        )
        v2 = assess_assistant_turn("try <tool_call>x</tool_call>", cfg, {})
        sev2 = {x.type: x.severity for x in v2.violations}
        assert sev2[VIOLATION_UNEXECUTED_PATTERN] == 2.5

    def test_locus_override(self):
        cfg = cast(
            InvalidActionPenaltyConfig,
            dict(_CFG, locus_overrides={VIOLATION_UNEXECUTED_PATTERN: LOCUS_REWARD}),
        )
        v = assess_assistant_turn("try <tool_call>x</tool_call>", cfg, {})
        types = {x.type: x for x in v.violations}
        assert types[VIOLATION_UNEXECUTED_PATTERN].locus == LOCUS_REWARD


# Structured tool-call schema validation (N3) — wires tool_protocol into the verdict
_SCHEMA_CFG: InvalidActionPenaltyConfig = {
    "enabled": True,
    "invalid_action_penalty": 0.5,
    "malformed_thinking_penalty": 0.25,
    "use_environment_flags": True,
    "enable_schema_validation": True,
    "thinking_tags": ["<think>", "</think>"],
}
_REGISTRY = ToolRegistry(
    tools=(
        make_tool(
            "run_shell",
            "Run a shell command",
            {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        ),
    )
)


def _tool_call(name: str, arguments: dict) -> str:
    import json

    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


class TestSchemaValidation:
    def test_valid_call_no_violation(self):
        v = assess_assistant_turn(
            _tool_call("run_shell", {"cmd": "ls"}), _SCHEMA_CFG, {}, registry=_REGISTRY
        )
        assert not v.invalid_action and not v.violations

    def test_unknown_tool_violation(self):
        v = assess_assistant_turn(
            _tool_call("nope", {}), _SCHEMA_CFG, {}, registry=_REGISTRY
        )
        assert v.invalid_action
        assert any(x.type == VIOLATION_SCHEMA_UNKNOWN_TOOL for x in v.violations)

    def test_bad_args_schema_violation(self):
        v = assess_assistant_turn(
            _tool_call("run_shell", {}), _SCHEMA_CFG, {}, registry=_REGISTRY
        )
        assert v.invalid_action
        sv = [x for x in v.violations if x.type == VIOLATION_SCHEMA_ARGS]
        assert sv and sv[0].locus == LOCUS_ADVANTAGE
        assert sv[0].detail  # carries the schema error reason

    def test_no_registry_means_no_schema_tier(self):
        # Same bad-args call without a registry: schema tier is skipped entirely.
        v = assess_assistant_turn(_tool_call("run_shell", {}), _SCHEMA_CFG, {})
        assert not any(
            x.type in (VIOLATION_SCHEMA_ARGS, VIOLATION_SCHEMA_UNKNOWN_TOOL)
            for x in v.violations
        )


class TestPenaltyStepScale:
    def test_no_schedule_is_identity(self):
        assert penalty_scale_at_step(_CFG, 100) == 1.0

    def test_constant(self):
        cfg = cast(
            InvalidActionPenaltyConfig,
            dict(_CFG, penalty_step_scale={"schedule": "constant", "start": 0.3}),
        )
        assert penalty_scale_at_step(cfg, 5) == 0.3
        assert penalty_scale_at_step(cfg, None) == 0.3

    def test_linear_interpolation_and_clamp(self):
        cfg = cast(
            InvalidActionPenaltyConfig,
            dict(
                _CFG,
                penalty_step_scale={
                    "schedule": "linear",
                    "start": 0.0,
                    "end": 1.0,
                    "steps": 10,
                },
            ),
        )
        assert penalty_scale_at_step(cfg, 0) == 0.0
        assert penalty_scale_at_step(cfg, 5) == 0.5
        assert penalty_scale_at_step(cfg, 10) == 1.0
        assert penalty_scale_at_step(cfg, 100) == 1.0  # clamped at end
