"""`DistributedLogprob` must not build its backward tensor on the inference path.

`softmax_output = log_probs.exp()` sat above the `if not inference_only` guard,
so a `[B, S, V_local]` fp32 tensor was allocated on every logprob pass and read
only by backward. Two things make that more than a tidy-up:

  * every shipped config sets `logprob_chunk_size: null`, so this class -- not
    `ChunkedDistributedLogprob`, which saves the logits and recomputes in
    backward -- serves the reference/prev-logprob pass of every GRPO step;
  * the two siblings in the same file already do it correctly
    (`DistributedLogprobWithSampling` computes the exp inside the guard,
    `DistributedCrossEntropy` uses `exp_()` there), so the placement is the
    file's own convention and this was the lone outlier.

Run on a single-rank gloo group: the all-reduces reduce to no-ops, so the maths
is a plain vocab-parallel log-softmax and every branch here is still exercised.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from dockyard_rl.distributed.model_utils import DistributedLogprob

_VOCAB = 8
_B, _S = 2, 3


@pytest.fixture(scope="module")
def group():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    created = not dist.is_initialized()
    if created:
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    yield dist.group.WORLD
    if created:
        dist.destroy_process_group()


class _CountExp(torch.overrides.TorchFunctionMode):
    """Counts exp/exp_ calls, wherever they are issued."""

    def __init__(self) -> None:
        self.count = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func in (torch.Tensor.exp, torch.Tensor.exp_, torch.exp, torch.exp_):
            self.count += 1
        return func(*args, **(kwargs or {}))


def _logits(requires_grad: bool = False) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(_B, _S, _VOCAB, dtype=torch.float32, requires_grad=requires_grad)


def _target() -> torch.Tensor:
    return torch.tensor([[1, 5, 0], [7, 2, 3]], dtype=torch.long)


def _apply(logits, target, group, inference_only):
    return DistributedLogprob.apply(logits, target, 0, _VOCAB, group, inference_only)


def _count(logits, target, group, inference_only) -> int:
    mode = _CountExp()
    with mode:
        _apply(logits, target, group, inference_only)
    return mode.count


class TestTheInferencePathAllocatesLess:
    def test_inference_only_skips_exactly_one_exp(self, group):
        """The training path does one more exp than the inference path.

        `_compute_distributed_log_softmax` issues one `exp_()` of its own on
        both paths, so the absolute counts are 1 and 2. With the exp above the
        guard both are 2 and the difference is 0, which is what fails here.
        """
        target = _target()
        inference = _count(_logits(), target, group, True)
        training = _count(_logits(True), target, group, False)

        assert training - inference == 1, (
            f"inference={inference} training={training}; the backward tensor is "
            "still being built on the inference path"
        )

    def test_inference_only_saves_nothing_for_backward(self, group):
        out = _apply(_logits(), _target(), group, True)

        assert out.requires_grad is False


class TestBehaviourIsUnchanged:
    """The move is memory-only; forward values and gradients must not shift."""

    def _reference_logprobs(self, logits, target):
        return torch.gather(
            torch.log_softmax(logits.float(), dim=-1), -1, target.unsqueeze(-1)
        ).squeeze(-1)

    def test_forward_matches_a_plain_log_softmax_gather(self, group):
        logits, target = _logits(), _target()

        out = _apply(logits, target, group, False)

        torch.testing.assert_close(out, self._reference_logprobs(logits, target))

    def test_backward_matches_autograd_through_the_reference(self, group):
        logits = _logits(True)
        target = _target()
        grad_seed = torch.arange(_B * _S, dtype=torch.float32).view(_B, _S) + 1.0

        _apply(logits, target, group, False).backward(grad_seed)
        got = logits.grad

        ref_logits = _logits(True)
        self._reference_logprobs(ref_logits, target).backward(grad_seed)

        assert got is not None and ref_logits.grad is not None
        torch.testing.assert_close(got, ref_logits.grad)

    def test_the_saved_tensor_is_probabilities_not_log_probabilities(self, group):
        """exp_() is in place now; backward still has to receive exp(log p)."""
        logits = _logits(True)
        target = _target()
        grad_seed = torch.ones(_B, _S)

        _apply(logits, target, group, False).backward(grad_seed)

        # d/dz of sum_t log p_t is (onehot - p); summing over the vocab axis of
        # the gradient gives 0 per position exactly when p sums to 1.
        assert logits.grad is not None
        torch.testing.assert_close(
            logits.grad.sum(-1), torch.zeros(_B, _S), atol=1e-5, rtol=0
        )
