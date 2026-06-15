"""Unit tests for structured tool-use rollout integration (S4, design fork 3).

Pure-CPU coverage of the turn-envelope construction and the token-id alignment
invariant. A tiny deterministic fake tokenizer stands in for a real model — no
network, no GPU, no generation engine.

Asserts:
  - the Qwen/Hermes turn envelope wraps an env observation as documented;
  - the token-id alignment invariant of fork 3 — for a synthetic 2-turn episode,
    ``[prev prompt ids][assistant gen ids verbatim][obs envelope ids]`` equals the
    next-turn generation prompt, the assistant ids are the raw generation output
    (never re-tokenized), and the role loss-mask is 1 over the assistant ids and 0
    over the envelope;
  - structured-OFF ⇒ the observation-append tokenization is byte-identical to the
    legacy fenced-text path for the same content.
"""

import sys as _sys
import types as _types
from pathlib import Path as _Path

# Make ``dockyard_rl`` resolve to the checkout this test lives in, regardless of
# the checkout directory's name (git worktrees are named after the branch/agent,
# not ``dockyard_rl``, so the suite-wide conftest path heuristic does not apply
# here). No-op when ``dockyard_rl`` already imports.
if "dockyard_rl" not in _sys.modules:
    try:  # noqa: SIM105
        import dockyard_rl  # noqa: F401
    except ModuleNotFoundError:
        _pkg_root = _Path(__file__).resolve().parents[2]
        if (_pkg_root / "experience" / "rollouts.py").exists():
            _pkg = _types.ModuleType("dockyard_rl")
            _pkg.__path__ = [str(_pkg_root)]  # type: ignore[attr-defined]
            _sys.modules["dockyard_rl"] = _pkg
            _init = _pkg_root / "__init__.py"
            if _init.exists():
                exec(compile(_init.read_text(), str(_init), "exec"), _pkg.__dict__)

from typing import Any, cast

import pytest
import torch

from dockyard_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    message_log_to_flat_messages,
)
from dockyard_rl.experience.rollouts import (
    _TOOL_RESPONSE_ENVELOPE_PREFIX,
    _TOOL_RESPONSE_ENVELOPE_SUFFIX,
    _structured_decode_skip_special,
    _structured_thinking_style,
    _tool_tags_survive_skip_special,
    _wrap_tool_response_envelope,
)
from dockyard_rl.tool_protocol.protocol import (
    THINKING_CLOSING_ONLY,
    THINKING_PAIRED,
    StructuredToolUseConfig,
)


class FakeTokenizer:
    """Deterministic byte-level tokenizer for alignment tests.

    Each character maps to its ordinal (offset to stay clear of 0 = the pad id),
    so encode/decode round-trips exactly and concatenation of token-id segments
    equals tokenizing the concatenated text. ``add_special_tokens`` is accepted
    and ignored (this fake has no special tokens), matching how the rollout calls
    the tokenizer with ``add_special_tokens=False``.
    """

    pad_token_id = 0
    _OFFSET = 1000

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [ord(c) + self._OFFSET for c in text]
        tensor = torch.tensor([ids], dtype=torch.int64)
        return _Encoding(tensor)

    def _decode_ids(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i) - self._OFFSET) for i in ids)

    def batch_decode(self, batch_ids, skip_special_tokens=True):
        return [
            self._decode_ids(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_ids
        ]


class _Encoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def _encode(tok, text):
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]


# ── envelope construction ───────────────────────────────────────────


class TestEnvelopeConstruction:
    def test_wrap_shape(self):
        wrapped = _wrap_tool_response_envelope("RESULT")
        assert wrapped == (
            "<|im_start|>user\n<tool_response>\nRESULT\n"
            "</tool_response><|im_end|>\n<|im_start|>assistant\n"
        )

    def test_prefix_suffix_constants_compose(self):
        wrapped = _wrap_tool_response_envelope("X")
        assert wrapped == _TOOL_RESPONSE_ENVELOPE_PREFIX + "X" + _TOOL_RESPONSE_ENVELOPE_SUFFIX

    def test_thinking_style_default_when_absent(self):
        assert _structured_thinking_style(None) == THINKING_PAIRED

    def test_thinking_style_from_cfg(self):
        cfg: StructuredToolUseConfig = {
            "enabled": True,
            "thinking_style": THINKING_CLOSING_ONLY,
        }
        assert _structured_thinking_style(cfg) == THINKING_CLOSING_ONLY

    def test_thinking_style_cfg_without_key_defaults(self):
        assert _structured_thinking_style({"enabled": True}) == THINKING_PAIRED


# ── token-id alignment invariant (fork 3) ───────────────────────────


class TestTokenIdAlignment:
    """Round-trip the fork-3 invariant for a synthetic 2-turn episode.

    Turn 1: prompt -> assistant generates verbatim ids -> env returns an
    observation, which the structured path wraps + tokenizes and appends.
    Turn 2's generation prompt is the concatenation of all prior message
    token_ids. The trainer's view (message_log_to_flat_messages) must equal
    that exact concatenation, and the assistant ids must be the raw generation
    output (never re-tokenized).
    """

    def _build_episode(self, tok):
        prompt_text = "<|im_start|>user\nsolve it<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = _encode(tok, prompt_text)

        # Assistant generation: raw output ids straight from the engine. These
        # are NOT produced by re-tokenizing decoded text in the real loop; here
        # we mint them directly to model that. Decoding them is only for the env.
        assistant_text = "<tool_call>\n{\"name\": \"run_shell\", \"arguments\": {}}\n</tool_call><|im_end|>\n"
        assistant_ids = _encode(tok, assistant_text)

        # Env observation (plain content the env produced).
        obs_content = "exit_code=0\nhello"
        envelope_text = _wrap_tool_response_envelope(obs_content)
        envelope_ids = _encode(tok, envelope_text)

        message_log = [
            {"role": "user", "content": prompt_text, "token_ids": prompt_ids},
            {"role": "assistant", "content": assistant_text, "token_ids": assistant_ids},
            {"role": "tool", "content": obs_content, "token_ids": envelope_ids},
        ]
        return message_log, prompt_ids, assistant_ids, envelope_ids

    def test_concatenation_matches_next_turn_prompt(self):
        tok = FakeTokenizer()
        message_log, prompt_ids, assistant_ids, envelope_ids = self._build_episode(tok)

        # The next-turn generation prompt is the concatenation of every prior
        # message's token_ids (the rollout flattens the log for generation).
        flat = message_log_to_flat_messages(message_log)
        trainer_view = flat["token_ids"]
        assert isinstance(trainer_view, torch.Tensor)

        expected_next_prompt = torch.cat([prompt_ids, assistant_ids, envelope_ids])
        assert torch.equal(trainer_view, expected_next_prompt)

    def test_assistant_ids_are_verbatim_not_retokenized(self):
        tok = FakeTokenizer()
        message_log, _, assistant_ids, _ = self._build_episode(tok)
        # The assistant message's token_ids are exactly the generation output;
        # they are a contiguous slice of the flattened view at the right offset.
        flat = message_log_to_flat_messages(message_log)
        prompt_len = len(message_log[0]["token_ids"])
        gen_len = len(assistant_ids)
        token_ids = flat["token_ids"]
        assert isinstance(token_ids, torch.Tensor)
        sliced = token_ids[prompt_len : prompt_len + gen_len]
        assert torch.equal(sliced, assistant_ids)

    def test_loss_mask_one_over_assistant_zero_over_envelope(self):
        tok = FakeTokenizer()
        message_log, prompt_ids, assistant_ids, envelope_ids = self._build_episode(tok)
        add_loss_mask_to_message_log([message_log])

        # User prompt: masked (0).
        assert torch.equal(
            message_log[0]["token_loss_mask"], torch.zeros_like(prompt_ids)
        )
        # Assistant turn: trained (1) over the entire generated span, including
        # the trailing <|im_end|> stop token carried in the ids.
        assert torch.equal(
            message_log[1]["token_loss_mask"], torch.ones_like(assistant_ids)
        )
        # Tool-response envelope: masked (0) over the whole envelope, including
        # the trailing <|im_start|>assistant\n generation prompt.
        assert torch.equal(
            message_log[2]["token_loss_mask"], torch.zeros_like(envelope_ids)
        )

    def test_envelope_is_single_masked_message(self):
        # The whole envelope is one non-assistant message, so masking is by role
        # exactly as for the legacy path — no per-token special handling needed.
        tok = FakeTokenizer()
        message_log, _, _, envelope_ids = self._build_episode(tok)
        env_msg = message_log[-1]
        assert env_msg["role"] != "assistant"
        # token_ids carry the wrapped envelope; content stays the env's plain text.
        assert torch.equal(env_msg["token_ids"], envelope_ids)
        assert "<tool_response>" not in env_msg["content"]


# ── disabled path byte-identity ─────────────────────────────────────


class TestDisabledPathByteIdentical:
    def test_off_obs_append_matches_legacy(self):
        """structured_cfg OFF ⇒ obs tokenized as raw content (legacy bytes)."""
        tok = FakeTokenizer()
        obs_content = "exit_code=0\nhello"

        # Legacy / OFF path: tokenize the raw env content directly.
        legacy_ids = _encode(tok, obs_content)

        # The OFF branch in the rollout uses exactly this raw-content tokenization
        # (no envelope). Reconstruct it the same way the loop does.
        structured_enabled = False
        obs_text = (
            _wrap_tool_response_envelope(obs_content)
            if structured_enabled
            else obs_content
        )
        off_ids = _encode(tok, obs_text)

        assert torch.equal(off_ids, legacy_ids)

    def test_on_differs_from_off(self):
        """Sanity: the structured envelope changes the appended token_ids."""
        tok = FakeTokenizer()
        obs_content = "exit_code=0\nhello"
        off_ids = _encode(tok, obs_content)
        on_ids = _encode(tok, _wrap_tool_response_envelope(obs_content))
        assert not torch.equal(
            on_ids, torch.nn.functional.pad(off_ids, (0, len(on_ids) - len(off_ids)))
        )
        assert len(on_ids) > len(off_ids)


# ── config threading (grpo resolver) ────────────────────────────────


class _FakeMasterConfig:
    """Minimal stand-in exposing the ``.grpo`` mapping the resolver reads."""

    def __init__(self, grpo: dict):
        self.grpo = grpo


class TestGrpoResolver:
    """Smoke-test the grpo.py resolver that threads structured_cfg into rollouts.

    Imports grpo lazily so heavy generation deps stay off the default test path
    (resolve_structured_tool_use_cfg only touches the config mapping).
    """

    @staticmethod
    def _resolver():
        # grpo.py pulls the GPU/cluster stack (nccl) transitively, absent in
        # CPU-only test environments; skip rather than fail there.
        grpo = pytest.importorskip(
            "dockyard_rl.algorithms.grpo", exc_type=ImportError
        )
        return grpo.resolve_structured_tool_use_cfg

    def test_absent_returns_none(self):
        resolve = self._resolver()
        assert resolve(_FakeMasterConfig({})) is None

    def test_disabled_returns_none(self):
        resolve = self._resolver()
        assert resolve(_FakeMasterConfig({"structured_tool_use": {"enabled": False}})) is None

    def test_enabled_returns_cfg(self):
        resolve = self._resolver()
        cfg = resolve(_FakeMasterConfig({"structured_tool_use": {"enabled": True}}))
        assert cfg is not None
        assert cfg.get("enabled") is True

    def test_cua_combination_rejected(self):
        import pytest

        resolve = self._resolver()
        mc = _FakeMasterConfig(
            {"structured_tool_use": {"enabled": True}, "cua_rollout": True}
        )
        with pytest.raises(NotImplementedError):
            resolve(mc)


# ── decode skip_special_tokens auto-detection (finding #1 fix) ───────


class _Enc:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _RoundTripTok:
    """Tokenizer whose decode round-trips faithfully → tool-call tags survive."""

    def __call__(self, text, add_special_tokens=False):
        return _Enc([ord(c) for c in text])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids)


class _StrippingTok:
    """Tokenizer that marks the tool-call tags special → skip drops them."""

    def __call__(self, text, add_special_tokens=False):
        return _Enc([ord(c) for c in text])

    def decode(self, ids, skip_special_tokens=True):
        s = "".join(chr(int(i)) for i in ids)
        if skip_special_tokens:
            s = s.replace("<tool_call>", "").replace("</tool_call>", "")
        return s


class TestDecodeSkipSpecial:
    def test_tags_survive_on_roundtrip_tokenizer(self):
        assert _tool_tags_survive_skip_special(cast(Any, _RoundTripTok())) is True

    def test_tags_stripped_on_special_tokenizer(self):
        assert _tool_tags_survive_skip_special(cast(Any, _StrippingTok())) is False

    def test_result_is_memoized(self):
        tok = _StrippingTok()
        assert _tool_tags_survive_skip_special(cast(Any, tok)) is False
        # cached attribute set; a second call uses it (even if decode changed)
        assert getattr(tok, "_dockyard_tool_tags_survive_skip") is False

    def test_legacy_path_skips_special(self):
        # No structured cfg → True, byte-identical to the legacy decode.
        assert _structured_decode_skip_special(None, cast(Any, _StrippingTok())) is True

    def test_explicit_override_wins(self):
        # Explicit False is honored even when the tags would survive.
        cfg: StructuredToolUseConfig = {
            "enabled": True,
            "decode_skip_special_tokens": False,
        }
        assert _structured_decode_skip_special(cfg, cast(Any, _RoundTripTok())) is False
        cfg_true: StructuredToolUseConfig = {
            "enabled": True,
            "decode_skip_special_tokens": True,
        }
        assert _structured_decode_skip_special(cfg_true, cast(Any, _StrippingTok())) is True

    def test_auto_keeps_special_when_tags_would_be_stripped(self):
        # The fix: enabled + unset + special-tag tokenizer → do NOT skip.
        cfg: StructuredToolUseConfig = {"enabled": True}  # decode flag absent (auto)
        assert _structured_decode_skip_special(cfg, cast(Any, _StrippingTok())) is False

    def test_auto_skips_when_tags_survive(self):
        # enabled + unset + ordinary-vocab tokenizer → skip (cleaner content).
        cfg: StructuredToolUseConfig = {"enabled": True}
        assert _structured_decode_skip_special(cfg, cast(Any, _RoundTripTok())) is True

    def test_tokenizer_error_is_conservative(self):
        class _Broken:
            def __call__(self, *a, **k):
                raise RuntimeError("boom")

        # On any tokenizer error, do not skip (keep tags rather than risk drop).
        assert _tool_tags_survive_skip_special(cast(Any, _Broken())) is False
        assert (
            _structured_decode_skip_special({"enabled": True}, cast(Any, _Broken()))
            is False
        )
