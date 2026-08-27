"""use_kl_in_reward must refuse the path where prev_logprobs is zeroed.

loss_fn.force_on_policy_ratio with no grpo.seq_logprob_error_threshold skips
the prev_logprobs forward and substitutes torch.zeros_like. Every
compute_advantage call site passes that column as logprobs_policy, so
ReinforcePlusPlus would take calculate_kl(zeros, reference) and subtract
kl_coef * that from every advantage. Shapes are right, so nothing errors.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from dockyard_rl.algorithms.grpo import _validate_kl_in_reward_compat


def _cfg(
    *,
    use_kl_in_reward=True,
    force_on_policy_ratio=True,
    seq_logprob_error_threshold=None,
    kl_penalty=0.01,
):
    return cast(Any, SimpleNamespace(
        loss_fn=SimpleNamespace(
            use_kl_in_reward=use_kl_in_reward,
            force_on_policy_ratio=force_on_policy_ratio,
            reference_policy_kl_penalty=kl_penalty,
        ),
        grpo={"seq_logprob_error_threshold": seq_logprob_error_threshold},
    ))


def test_the_corrupting_combination_is_refused():
    with pytest.raises(ValueError) as exc:
        _validate_kl_in_reward_compat(_cfg())
    msg = str(exc.value)
    assert "use_kl_in_reward" in msg
    assert "force_on_policy_ratio" in msg
    # the message must name every way out, not just the flag it refuses
    assert "seq_logprob_error_threshold" in msg
    assert "reference_policy_kl_penalty" in msg


@pytest.mark.parametrize(
    "kwargs,why",
    [
        ({"use_kl_in_reward": False}, "KL term is off"),
        ({"force_on_policy_ratio": False}, "prev_logprobs is really computed"),
        ({"seq_logprob_error_threshold": 0.1}, "threshold keeps the forward"),
        ({"kl_penalty": 0.0}, "kl_coef zeroes the term regardless"),
    ],
)
def test_every_safe_combination_is_allowed(kwargs, why):
    _validate_kl_in_reward_compat(_cfg(**kwargs))  # must not raise


def test_the_guard_is_wired_into_setup():
    """Wire cut: drop the call from setup() and this goes red."""
    import inspect

    from dockyard_rl.algorithms import grpo

    setup_src = inspect.getsource(grpo.setup)
    assert "_validate_kl_in_reward_compat(master_config)" in setup_src, (
        "the refusal exists but setup() never calls it, so a corrupting "
        "config still starts"
    )
