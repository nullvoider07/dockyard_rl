"""Consolidation: the JAX backend is wired end-to-end (config -> selection -> actor).

Loads the shipped JAX example config through the real OmegaConf path and checks
the selection plumbing that the driver relies on, plus that the worker FQN
resolves to a Ray actor class (the RayWorkerBuilder requires `.options/.remote`).
"""

from __future__ import annotations

import importlib
from typing import Any, cast

import pytest

pytest.importorskip("jax")
from omegaconf import OmegaConf

from dockyard_rl.models.jax.config import jax_backend_enabled, resolve_trainer_env
from dockyard_rl.utils.config import load_config, register_omegaconf_resolvers

_CONFIG = "examples/configs/grpo_swe_jax.yaml"


def _policy() -> dict[str, Any]:
    register_omegaconf_resolvers()
    cfg = load_config(_CONFIG)
    container = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True))
    return cast("dict[str, Any]", container["policy"])


def test_jax_config_selects_jax_backend():
    pol = _policy()
    assert jax_backend_enabled(pol) is True
    assert "dtensor_cfg" not in pol  # JAX config carries no dtensor block
    # interpolation resolved against jax_cfg, not dtensor_cfg
    assert pol["make_sequence_length_divisible_by"] == pol["jax_cfg"]["tensor_parallel_size"]
    # JAX worker requires plain microbatching (no packing / dynamic batching)
    assert pol["sequence_packing"]["enabled"] is False
    assert pol["dynamic_batching"]["enabled"] is False


def test_trainer_env_has_prealloc_false():
    env = resolve_trainer_env(_policy()["jax_cfg"])
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_worker_fqn_is_ray_actor():
    # The FQN lm_policy selects must be a Ray actor (has .options/.remote).
    mod = importlib.import_module("dockyard_rl.models.jax.policy_worker")
    actor = mod.JaxPolicyWorker
    assert hasattr(actor, "options") and hasattr(actor, "remote")
    # The plain impl is constructible directly (the actor is not).
    assert isinstance(mod.JaxPolicyWorkerImpl, type)
