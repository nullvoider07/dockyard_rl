"""CPU tests for the cross-tokenizer CP/TP primitives.

The cross-tokenizer loss body and the teacher-logit transport keystone invoke
these unconditionally, so they must collapse to the plain local op at world<=1
(single process / CPU). These tests pin that collapse plus the gradient flow of
the differentiable all-reduce; the multi-rank behavior is GPU-deferred.
"""

import torch

from dockyard_rl.distributed.model_utils import (
    cp_load_balanced_to_contiguous,
    cp_shift_next,
    group_all_reduce_sum,
    group_all_reduce_sum_with_grad,
    vocab_parallel_argmax,
    vocab_parallel_log_softmax,
)


# -- group_all_reduce_sum[_with_grad] -----------------------------------------

def test_group_all_reduce_sum_is_value_noop_at_world1():
    x = torch.randn(3, 4)
    out = group_all_reduce_sum(x, None)
    assert torch.equal(out, x)


def test_group_all_reduce_sum_no_grad():
    x = torch.randn(2, 2, requires_grad=True)
    out = group_all_reduce_sum(x, None)
    assert not out.requires_grad


def test_group_all_reduce_sum_with_grad_passes_gradient():
    x = torch.randn(2, 3, requires_grad=True)
    out = group_all_reduce_sum_with_grad(x, None)
    assert torch.equal(out, x)
    out.sum().backward()
    # backward is identity: d(sum)/dx = ones.
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x))


# -- cp_load_balanced_to_contiguous -------------------------------------------

def test_cp_load_balanced_to_contiguous_identity_at_world1():
    x = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    assert cp_load_balanced_to_contiguous(x, cp_group=None) is x


# -- cp_shift_next ------------------------------------------------------------

def test_cp_shift_next_local_left_roll_with_fill():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    out = cp_shift_next(x, None, fill=-1.0)
    assert out.tolist() == [[2.0, 3.0, 4.0, -1.0], [6.0, 7.0, 8.0, -1.0]]


def test_cp_shift_next_fill_sentinel_for_chunk_ids():
    # chunk-id use: long tensor, fill=-1 marks "no next chunk" at the boundary.
    x = torch.tensor([[0, 0, 1, 2]])
    out = cp_shift_next(x, None, fill=-1)
    assert out.tolist() == [[0, 1, 2, -1]]


# -- vocab_parallel_log_softmax -----------------------------------------------

def test_vocab_parallel_log_softmax_matches_local_with_temperature():
    logits = torch.randn(2, 4, 7)
    T = 1.5
    out = vocab_parallel_log_softmax(logits, T)
    expected = torch.log_softmax(logits.float() / T, dim=-1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_vocab_parallel_log_softmax_is_differentiable():
    logits = torch.randn(1, 3, 5, requires_grad=True)
    out = vocab_parallel_log_softmax(logits, 1.0)
    out.sum().backward()
    assert logits.grad is not None


# -- vocab_parallel_argmax ----------------------------------------------------

def test_vocab_parallel_argmax_matches_local_argmax():
    logits = torch.randn(2, 5, 9)
    out = vocab_parallel_argmax(logits)
    assert out.shape == (2, 5)
    assert torch.equal(out, logits.argmax(dim=-1))


def test_vocab_parallel_argmax_no_grad():
    logits = torch.randn(1, 2, 4, requires_grad=True)
    out = vocab_parallel_argmax(logits)
    assert not out.requires_grad
