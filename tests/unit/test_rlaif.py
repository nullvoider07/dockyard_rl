"""Unit tests for RLAIF / Constitutional preference-data generation (D6).

CPU coverage of the critique→revise pipeline, constitutional/abstention pair
construction, and the flow into the existing preference collate — all driven by
a stub chat LLM and a fake tokenizer. Live model/judge generation is HV.
"""

import pytest
import torch

from dockyard_rl.algorithms.rlaif import (
    Constitution,
    ConstitutionPrinciple,
    DEFAULT_CONSTITUTION,
    build_abstention_pair,
    build_constitutional_pair,
    build_rlaif_preference_data,
    generate_constitutional_revision,
)
from typing import cast
from transformers import PreTrainedTokenizerBase
from dockyard_rl.data.collate_fn import preference_collate_fn


class _FakeTokenizer:
    pad_token_id = 0

    def encode(self, text):
        # Deterministic token ids from word count (≥1 token).
        n = max(1, len(text.split()))
        return list(range(1, n + 1))


class _ScriptedLLM:
    """Returns canned replies in sequence; records calls."""

    enabled = True

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def chat(self, messages: list[dict], *, temperature: float = 0) -> str | None:
        self.calls.append(messages[0]["content"])
        if not self._replies:
            return None
        return self._replies.pop(0)


_ONE_PRINCIPLE = Constitution(
    [ConstitutionPrinciple("p", "critique it", "revise it")]
)


class TestConstitution:
    def test_empty_constitution_rejected(self):
        with pytest.raises(ValueError):
            Constitution([])

    def test_default_has_principles(self):
        assert len(DEFAULT_CONSTITUTION.principles) >= 3


class TestRevisionPipeline:
    def test_single_principle_revises(self):
        llm = _ScriptedLLM(["this is bad because X", "a better answer"])
        revised, trace = generate_constitutional_revision(
            llm, "prompt", "bad answer", _ONE_PRINCIPLE
        )
        assert revised == "a better answer"
        assert len(trace) == 1
        assert trace[0]["principle"] == "p"

    def test_no_revision_marker_skips(self):
        llm = _ScriptedLLM(["NO REVISION NEEDED"])
        revised, trace = generate_constitutional_revision(
            llm, "prompt", "fine answer", _ONE_PRINCIPLE
        )
        assert revised == "fine answer"
        assert trace == []

    def test_judge_failure_skips_principle(self):
        llm = _ScriptedLLM([None])  # critique call fails
        revised, trace = generate_constitutional_revision(
            llm, "prompt", "answer", _ONE_PRINCIPLE
        )
        assert revised == "answer" and trace == []

    def test_multiple_principles_thread_response(self):
        # critique1, revise1, critique2, revise2
        llm = _ScriptedLLM(["c1", "rev1", "c2", "rev2"])
        constitution = Constitution(
            [
                ConstitutionPrinciple("a", "ca", "ra"),
                ConstitutionPrinciple("b", "cb", "rb"),
            ]
        )
        revised, trace = generate_constitutional_revision(
            llm, "prompt", "orig", constitution
        )
        assert revised == "rev2"
        assert [t["principle"] for t in trace] == ["a", "b"]

    def test_max_principles_caps(self):
        llm = _ScriptedLLM(["c1", "rev1", "c2", "rev2"])
        revised, trace = generate_constitutional_revision(
            llm, "prompt", "orig", DEFAULT_CONSTITUTION, max_principles=1
        )
        assert len(trace) == 1


class TestConstitutionalPair:
    def test_builds_pair_with_revision(self):
        llm = _ScriptedLLM(["critique", "revised text here"])
        pair = build_constitutional_pair(
            "the prompt", "original response", llm, _FakeTokenizer(), _ONE_PRINCIPLE
        )
        assert pair is not None
        chosen, rejected = pair
        assert chosen.response_text == "revised text here"
        assert rejected.response_text == "original response"
        assert isinstance(chosen.response_token_ids, torch.Tensor)

    def test_no_pair_when_no_revision(self):
        llm = _ScriptedLLM(["NO REVISION NEEDED"])
        pair = build_constitutional_pair(
            "prompt", "good answer", llm, _FakeTokenizer(), _ONE_PRINCIPLE
        )
        assert pair is None

    def test_no_pair_when_revision_equals_original(self):
        llm = _ScriptedLLM(["critique", "  same answer  "])
        pair = build_constitutional_pair(
            "prompt", "same answer", llm, _FakeTokenizer(), _ONE_PRINCIPLE
        )
        assert pair is None  # revised strips-equal to original


class TestAbstentionPair:
    def test_chosen_is_abstention(self):
        chosen, rejected = build_abstention_pair(
            "who won the 2050 election?", "Candidate X won decisively.",
            _FakeTokenizer(),
        )
        assert "not certain" in chosen.response_text.lower()
        assert rejected.response_text == "Candidate X won decisively."

    def test_custom_abstention_text(self):
        chosen, _ = build_abstention_pair(
            "p", "wrong", _FakeTokenizer(), abstention_text="I don't know."
        )
        assert chosen.response_text == "I don't know."


class TestRlaifPipeline:
    def test_build_preference_data_and_collate(self):
        # Two inputs: one revises, one does not.
        llm = _ScriptedLLM(["crit", "better", "NO REVISION NEEDED"])
        data, metrics = build_rlaif_preference_data(
            [("p1", "bad1"), ("p2", "fine2")],
            llm, _FakeTokenizer(), _ONE_PRINCIPLE,
        )
        assert metrics["num_inputs"] == 2
        assert metrics["num_pairs"] == 1
        assert metrics["num_no_revision"] == 1
        # Flows through the real preference collate (1 pair → batch of 2).
        batch = preference_collate_fn(
            data, tokenizer=cast(PreTrainedTokenizerBase, _FakeTokenizer()),
            make_sequence_length_divisible_by=1, add_loss_mask=True,
        )
        assert batch["input_ids"].shape[0] == 2
        assert "token_mask" in batch

    def test_idx_offset(self):
        llm = _ScriptedLLM(["c", "rev"])
        data, _ = build_rlaif_preference_data(
            [("p", "orig")], llm, _FakeTokenizer(), _ONE_PRINCIPLE, start_idx=5
        )
        assert data[0]["idx"] == 5
