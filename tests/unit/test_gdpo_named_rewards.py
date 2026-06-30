"""CPU tests for the GDPO named multi-reward representation.

Multi-reward environments expose components under namespaced "reward/<name>" keys
(instead of positional reward1/reward2/...), and EnvironmentReturn.rewards is a
dict[str, Tensor] for those envs. These cover the key-derivation helper, the
GDPOAdvantageEstimator consuming named keys, and the producer-side dict
accumulation + mixed-format guard in calculate_rewards.
"""

from typing import Any, cast

import pytest
import torch

from dockyard_rl.algorithms.advantage_estimator import GDPOAdvantageEstimator
from dockyard_rl.algorithms.loss.loss_functions import ClippedPGLossConfig
from dockyard_rl.algorithms.utils import get_gdpo_reward_component_keys
from dockyard_rl.environments.interfaces import EnvironmentReturn
from dockyard_rl.experience import rollouts


# -- get_gdpo_reward_component_keys -------------------------------------------

def test_named_reward_keys_detected_and_sorted():
    batch = {
        "reward/format": torch.zeros(2),
        "reward/correctness": torch.zeros(2),
        "total_reward": torch.zeros(2),
        "input_ids": torch.zeros(2, 3),
    }
    # Only reward/* keys, sorted; non-reward keys excluded.
    assert get_gdpo_reward_component_keys(batch) == [
        "reward/correctness",
        "reward/format",
    ]


def test_legacy_positional_keys_are_not_matched():
    # Old reward1/reward2 scheme has no "reward/" prefix -> ignored.
    batch = {"reward1": torch.zeros(2), "reward2": torch.zeros(2), "reward": 1}
    assert get_gdpo_reward_component_keys(batch) == []


# -- GDPOAdvantageEstimator consumes named keys -------------------------------

def _estimator(normalize=False):
    return GDPOAdvantageEstimator(
        {"use_leave_one_out_baseline": True, "normalize_rewards": normalize},
        cast(ClippedPGLossConfig, {}),
    )


def test_gdpo_estimator_consumes_named_components():
    est = _estimator()
    prompt_ids = torch.tensor([[0], [0], [1], [1]])  # (batch, prompt_len)
    mask = torch.ones(4, 5)
    repeated_batch = {
        "reward/correctness": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "reward/format": torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }
    adv = est.compute_advantage(prompt_ids, None, mask, repeated_batch)
    assert adv.shape == mask.shape
    assert torch.isfinite(adv).all()


def test_gdpo_estimator_requires_two_named_components():
    est = _estimator()
    mask = torch.ones(2, 3)
    with pytest.raises(ValueError, match=r"reward/name1, reward/name2"):
        est.compute_advantage(
            torch.tensor([[0], [1]]), None, mask,
            {"reward/correctness": torch.tensor([1.0, 0.0])},
        )


# -- calculate_rewards producer path (dict accumulation + mixed-format guard) --

class _Call:
    """Stand-in for a Ray ObjectRef; carries its precomputed result."""

    def __init__(self, result):
        self.result = result


class _FakeStep:
    def __init__(self, make_result):
        self._make_result = make_result

    def remote(self, messages, env_info):
        return _Call(self._make_result(len(messages)))


class _FakeEnv:
    def __init__(self, make_result):
        self.step = _FakeStep(make_result)


def _dict_reward_env(n):
    return EnvironmentReturn(
        observations=[{"role": "environment", "content": "c"} for _ in range(n)],
        metadata=[{} for _ in range(n)],
        next_stop_strings=cast(Any, None),  # exercise the None-fill branch
        rewards={
            "reward/correctness": torch.arange(n, dtype=torch.float32),
            "reward/format": torch.ones(n, dtype=torch.float32),
        },
        terminateds=torch.ones(n),
        answers=None,
    )


def _scalar_reward_env(n):
    return EnvironmentReturn(
        observations=[{"role": "environment", "content": "c"} for _ in range(n)],
        metadata=[{} for _ in range(n)],
        next_stop_strings=cast(Any, None),  # exercise the None-fill branch
        rewards=torch.ones(n, dtype=torch.float32),
        terminateds=torch.ones(n),
        answers=None,
    )


def _batch(task_names):
    # calculate_rewards is typed for BatchedDataDict[DatumSpec]; a plain dict is
    # structurally sufficient for the fields it reads here (cast to satisfy pyright).
    n = len(task_names)
    return cast(Any, {
        "message_log": [[{"role": "user", "content": "q"}] for _ in range(n)],
        "task_name": task_names,
        "extra_env_info": [{} for _ in range(n)],
    })


def _envs(mapping):
    # Fake envs stand in for EnvironmentInterface ray actors.
    return cast(Any, mapping)


def _patch_ray_get(monkeypatch):
    monkeypatch.setattr(
        rollouts.ray, "get", lambda futures: [f.result for f in futures]
    )


def test_calculate_rewards_builds_named_dict(monkeypatch):
    _patch_ray_get(monkeypatch)
    task_to_env = _envs({"math": _FakeEnv(_dict_reward_env)})
    out = rollouts.calculate_rewards(_batch(["math", "math", "math"]), task_to_env)
    assert isinstance(out.rewards, dict)
    assert set(out.rewards) == {"reward/correctness", "reward/format"}
    # single group -> order preserved.
    assert torch.equal(out.rewards["reward/correctness"], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(out.rewards["reward/format"], torch.ones(3))


def test_calculate_rewards_rejects_mixed_dict_and_scalar(monkeypatch):
    _patch_ray_get(monkeypatch)
    task_to_env = _envs({
        "a": _FakeEnv(_dict_reward_env),
        "b": _FakeEnv(_scalar_reward_env),
    })
    with pytest.raises(AssertionError, match="Mixing dict-based and scalar rewards"):
        rollouts.calculate_rewards(_batch(["a", "b"]), task_to_env)


def test_calculate_rewards_single_reward_stays_tensor(monkeypatch):
    _patch_ray_get(monkeypatch)
    task_to_env = _envs({"a": _FakeEnv(_scalar_reward_env)})
    out = rollouts.calculate_rewards(_batch(["a", "a"]), task_to_env)
    assert isinstance(out.rewards, torch.Tensor)
    assert out.rewards.shape == (2,)
