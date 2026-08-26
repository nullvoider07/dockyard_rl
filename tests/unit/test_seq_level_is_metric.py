"""#3753: the sequence-level importance-ratio metric was inflated ~B-fold.

ClippedPGLossFn unsqueezes actor_importance_weights to [B, 1] on the
sequence-level path so it broadcasts against the [B, S] clip_loss. The metric
at the end of the call then passed that [B, 1] tensor to masked_mean with a
[B] sample_mask; masked_mean multiplies its two arguments, so the product was
[B, B] and .sum() covered B^2 terms while the divisor stayed global_valid_seqs.

Metric only -- the loss and the gradient consume the [B, 1] shape correctly,
which is exactly why it survived: tests/unit/test_jax_loss_parity.py compares
loss and gradient and never metrics, and its config matrix does not set
sequence_level_importance_ratios at all.

The JAX backend was already correct (models/jax/loss.py indexes actor_iw[:, 0]),
so this was a live torch/JAX divergence on a shared metric key.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dockyard_rl.algorithms.loss.loss_functions import (
    ClippedPGLossConfig,
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict


def _cfg(**kw) -> ClippedPGLossConfig:
    base: dict = dict(
        token_level_loss=False,  # mutually exclusive with seq-level IS
        ratio_clip_min=0.2,
        ratio_clip_max=0.2,
        reference_policy_kl_penalty=0.0,
        sequence_level_importance_ratios=True,
        use_importance_sampling_correction=True,
    )
    base.update(kw)
    return ClippedPGLossConfig(**base)


def _run(B: int = 4, S: int = 6, seed: int = 0, **cfg_kw):
    """Return (metrics, expected_ratio) for a seq-level-IS step.

    expected_ratio is computed independently of the loss: the per-sequence
    weight exp(sum over masked tokens of prev - gen), averaged over the valid
    samples with the same global normalization the loss uses.
    """
    rng = np.random.default_rng(seed)
    prev = rng.standard_normal((B, S)).astype(np.float32) * 0.3
    gen = rng.standard_normal((B, S)).astype(np.float32) * 0.3
    curr = rng.standard_normal((B, S - 1)).astype(np.float32) * 0.3
    adv = rng.standard_normal((B, S)).astype(np.float32)
    token_mask = np.ones((B, S), dtype=np.float32)
    token_mask[:, 0] = 0.0  # column 0 is sliced off inside the loss
    sample_mask = np.ones(B, dtype=np.float32)
    sample_mask[-1] = 0.0  # one invalid sample, so the mask is load-bearing

    data: BatchedDataDict[ClippedPGLossDataDict] = BatchedDataDict({
        "prev_logprobs": torch.from_numpy(prev),
        "generation_logprobs": torch.from_numpy(gen),
        "reference_policy_logprobs": torch.from_numpy(np.zeros_like(prev)),
        "advantages": torch.from_numpy(adv),
        "token_mask": torch.from_numpy(token_mask),
        "sample_mask": torch.from_numpy(sample_mask),
    })

    mask = token_mask[:, 1:] * sample_mask[:, None]
    gvs = torch.tensor(float(sample_mask.sum()))
    gvt = torch.tensor(float(mask.sum()))

    curr_t = torch.from_numpy(curr).clone().requires_grad_(True)
    _, metrics = ClippedPGLossFn(_cfg(**cfg_kw))(curr_t, data, gvs, gvt)

    seq_w = np.exp(((prev[:, 1:] - gen[:, 1:]) * mask).sum(axis=-1))
    expected = float((seq_w * sample_mask).sum() / sample_mask.sum())
    return metrics, expected


class TestSequenceImportanceRatioMetric:
    def test_metric_is_the_per_sequence_mean(self):
        """The load-bearing assertion: restore the [B, 1] shape and this fails."""
        metrics, expected = _run()
        assert metrics["sampling_importance_ratio"] == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize("B", [2, 4, 8])
    def test_metric_does_not_scale_with_batch_size(self, B):
        """The old shape summed B^2 terms over B, so the error grew with B.

        A single-B test would pass with a coincidental constant; sweeping B is
        what distinguishes a correct mean from an inflated one.
        """
        metrics, expected = _run(B=B)
        assert metrics["sampling_importance_ratio"] == pytest.approx(expected, rel=1e-5)

    def test_token_level_path_is_unchanged(self):
        """The non-seq-level branch was always [B, S] x [B, S] and stays put."""
        rng = np.random.default_rng(1)
        B, S = 4, 6
        prev = rng.standard_normal((B, S)).astype(np.float32) * 0.3
        gen = rng.standard_normal((B, S)).astype(np.float32) * 0.3
        curr = rng.standard_normal((B, S - 1)).astype(np.float32) * 0.3
        token_mask = np.ones((B, S), dtype=np.float32)
        token_mask[:, 0] = 0.0
        sample_mask = np.ones(B, dtype=np.float32)

        data: BatchedDataDict[ClippedPGLossDataDict] = BatchedDataDict({
            "prev_logprobs": torch.from_numpy(prev),
            "generation_logprobs": torch.from_numpy(gen),
            "reference_policy_logprobs": torch.from_numpy(np.zeros_like(prev)),
            "advantages": torch.from_numpy(rng.standard_normal((B, S)).astype(np.float32)),
            "token_mask": torch.from_numpy(token_mask),
            "sample_mask": torch.from_numpy(sample_mask),
        })
        mask = token_mask[:, 1:] * sample_mask[:, None]
        gvt = torch.tensor(float(mask.sum()))
        curr_t = torch.from_numpy(curr).clone().requires_grad_(True)

        cfg = ClippedPGLossConfig(
            token_level_loss=True,
            ratio_clip_min=0.2,
            ratio_clip_max=0.2,
            reference_policy_kl_penalty=0.0,
            sequence_level_importance_ratios=False,
            use_importance_sampling_correction=True,
        )
        _, metrics = ClippedPGLossFn(cfg)(
            curr_t, data, torch.tensor(float(sample_mask.sum())), gvt
        )

        tok_w = np.exp(prev[:, 1:] - gen[:, 1:])
        expected = float((tok_w * mask).sum() / mask.sum())
        assert metrics["sampling_importance_ratio"] == pytest.approx(expected, rel=1e-5)

    def test_gradient_path_still_sees_the_broadcast_shape(self):
        """The squeeze is metric-local; cutting it into the loss would show here.

        prev is set equal to curr so every sequence ratio is exactly 1 and no
        sample is PPO-clip-saturated -- otherwise a zero row means "clamped",
        not "unwired", and the assertion would prove nothing. With the clip
        inactive, a non-zero gradient on every row shows the [B, 1] weight
        reached the [B, S] clip_loss. Squeezing it there instead would make the
        multiply [B, S] * [B], which raises unless S happens to equal B.
        """
        rng = np.random.default_rng(2)
        B, S = 3, 5
        curr = rng.standard_normal((B, S - 1)).astype(np.float32) * 0.3
        prev = np.concatenate([np.zeros((B, 1), np.float32), curr], axis=1)
        gen = prev.copy()
        token_mask = np.ones((B, S), dtype=np.float32)
        token_mask[:, 0] = 0.0
        sample_mask = np.ones(B, dtype=np.float32)

        data: BatchedDataDict[ClippedPGLossDataDict] = BatchedDataDict({
            "prev_logprobs": torch.from_numpy(prev),
            "generation_logprobs": torch.from_numpy(gen),
            "reference_policy_logprobs": torch.from_numpy(np.zeros_like(prev)),
            "advantages": torch.from_numpy(np.ones((B, S), dtype=np.float32)),
            "token_mask": torch.from_numpy(token_mask),
            "sample_mask": torch.from_numpy(sample_mask),
        })
        mask = token_mask[:, 1:] * sample_mask[:, None]
        curr_t = torch.from_numpy(curr).clone().requires_grad_(True)
        loss, metrics = ClippedPGLossFn(_cfg())(
            curr_t,
            data,
            torch.tensor(float(sample_mask.sum())),
            torch.tensor(float(mask.sum())),
        )
        loss.backward()
        assert curr_t.grad is not None
        assert curr_t.grad.shape == (B, S - 1)
        assert (np.abs(curr_t.grad.numpy()).sum(axis=-1) > 0).all()
        # prev == gen makes every sequence weight exp(0) == 1, so the metric
        # pins to 1.0 independently of the shape arithmetic above.
        assert metrics["sampling_importance_ratio"] == pytest.approx(1.0, rel=1e-5)
