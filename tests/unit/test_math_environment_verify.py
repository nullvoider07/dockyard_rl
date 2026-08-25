"""HFVerifyWorker.verify score/answer alignment (#3635).

The defect: the score was appended mid-`try`, and the answer-extraction block
that followed could raise (three asserts, plus IndexError on an empty
prediction list). Control then fell into the `except`, which appended a
*second* score for the same sample. `results` grew longer than the batch and
rewards misaligned across every sample after the failing one — silently, since
nothing errors.

These are alignment tests first and content tests second. The batch-length
invariant is the one that matters: it fails if the commit-once structure is
reverted, even though the extraction helper is independently total.

No Ray is required — the actor class is unwrapped and __init__ bypassed, since
verify() reads only self.verify_func.
"""

from unittest.mock import patch

import pytest

from dockyard_rl.environments.math_environment import (
    HFVerifyWorker,
    _select_extracted_answer,
)

# ray decorates the class into an ActorClass; the original is reachable via
# __ray_metadata__, which ray does not declare on its public type.
_WorkerClass = HFVerifyWorker.__ray_metadata__.modified_class  # type: ignore[attr-defined]


def _worker(verify_func):
    w = object.__new__(_WorkerClass)
    w.verify_func = verify_func
    return w


class TestAlignment:
    """len(results) == len(batch), on every path."""

    def test_batch_length_holds_when_extraction_raises(self):
        # Force a raise *after* the score is computed. The commit-once
        # structure is what keeps the lists aligned here; the helper being
        # total is a separate, independent guard.
        w = _worker(lambda gold, pred: (1.0, (["42"], ["42"])))
        with patch(
            "dockyard_rl.environments.math_environment._select_extracted_answer",
            side_effect=RuntimeError("extraction blew up"),
        ):
            scores, answers = w.verify(
                ["a", "b", "c"], ["42", "42", "42"], return_extracted_answer=True
            )

        assert len(scores) == 3, f"results desynced from the batch: {scores}"
        assert len(answers) == 3

    def test_batch_length_holds_when_scoring_raises(self):
        def boom(gold, pred):
            raise ValueError("verifier exploded")

        w = _worker(boom)
        scores, answers = w.verify(
            ["a", "b"], ["1", "2"], return_extracted_answer=True
        )

        assert scores == [0.0, 0.0]
        assert answers == [None, None]

    def test_mixed_batch_stays_positionally_aligned(self):
        """A failure in the middle must not shift later samples' scores."""
        calls = {"n": 0}

        def flaky(gold, pred):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ValueError("second sample fails")
            return (1.0, (["7"], ["7"]))

        w = _worker(flaky)
        scores, _ = w.verify(
            ["a", "b", "c"], ["7", "7", "7"], return_extracted_answer=True
        )

        assert scores == [1.0, 0.0, 1.0], "middle failure shifted the batch"


class TestImplSelection:
    """The declared default must actually be reachable (#3635 neighbour)."""

    def test_absent_kwarg_uses_the_hf_default(self):
        w = _worker(lambda gold, pred: (1.0, (["42"], ["42"])))
        assert w.verify(["a"], ["42"]) == [1.0]

    def test_explicit_none_uses_the_hf_default(self):
        """math_verify_impl is NotRequired[str|None]; null must not zero the batch."""
        w = _worker(lambda gold, pred: (1.0, (["42"], ["42"])))
        assert w.verify(["a"], ["42"], math_verify_impl=None) == [1.0]

    def test_unknown_impl_still_scores_zero(self):
        w = _worker(lambda gold, pred: (1.0, (["42"], ["42"])))
        assert w.verify(["a"], ["42"], math_verify_impl="nope") == [0.0]


class TestExtractedAnswer:
    def test_records_first_prediction_not_its_first_character(self):
        """`extracted_prediction[0][0]` took the first *character* ("42" -> "4")."""
        # No gold match, so the fallback path runs.
        w = _worker(lambda gold, pred: (0.0, (["99"], ["42", "43"])))
        with patch(
            "dockyard_rl.environments.math_environment.grader.verify",
            return_value=False,
        ):
            _, answers = w.verify(["a"], ["99"], return_extracted_answer=True)

        assert answers == ["42"], "recorded a character slice, not the prediction"

    def test_score_survives_a_failed_extraction(self):
        """Extraction feeds cons@k, not the reward; it must not zero a pass."""
        w = _worker(lambda gold, pred: (1.0, "not-a-pair"))
        scores, answers = w.verify(["a"], ["42"], return_extracted_answer=True)

        assert scores == [1.0]
        assert answers == [None]

    def test_scores_only_mode_still_returns_a_bare_list(self):
        w = _worker(lambda gold, pred: (1.0, (["42"], ["42"])))
        out = w.verify(["a"], ["42"], return_extracted_answer=False)
        assert out == [1.0]


class TestSelectExtractedAnswerIsTotal:
    """The helper returns None for every malformed shape rather than raising."""

    @pytest.mark.parametrize(
        "bad", [None, "x", (), ("a",), ("a", "b", "c"), 42, ([], []), object()]
    )
    def test_malformed_shapes_return_none(self, bad):
        assert _select_extracted_answer(bad, "hf_math_verify") is None

    def test_a_throwing_grader_returns_none(self):
        with patch(
            "dockyard_rl.environments.math_environment.grader.verify",
            side_effect=RuntimeError("grader exploded"),
        ):
            assert (
                _select_extracted_answer((["1"], ["2"]), "hf_math_verify") is None
            )

    def test_dapo_branch_takes_the_plain_string(self):
        assert _select_extracted_answer("42", "dapo_math_verify") == "42"
        assert _select_extracted_answer(None, "dapo_math_verify") is None

    def test_gold_match_wins_over_the_first_prediction(self):
        with patch(
            "dockyard_rl.environments.math_environment.grader.verify",
            side_effect=lambda gold, pred: pred == "43",
        ):
            got = _select_extracted_answer((["43"], ["42", "43"]), "hf_math_verify")
        assert got == "43"
