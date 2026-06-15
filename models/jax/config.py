"""Configuration schema and trainer-fleet env for the JAX policy backend.

The ``policy.jax_cfg`` block mirrors ``policy.dtensor_cfg``: it selects the JAX
trainer worker and carries the mesh degrees and per-worker env. Keys are
intentionally a superset-compatible shape with ``dtensor_cfg`` (``enabled``,
``tensor_parallel_size``, ``context_parallel_size``, ``env_vars``) so the
driver-side selection logic in ``models/policy/lm_policy.py`` reads them the
same way for both backends.
"""

from __future__ import annotations

from typing import Any, Mapping, TypedDict

# Env vars the JAX trainer worker must run with. XLA preallocates ~75% of
# device memory by default; with torch (refit tensors) and potentially vLLM
# co-resident on the trainer device, that starves them. Disabling preallocation
# lets XLA grow on demand and leaves headroom for the torch-side refit path
# (Seam 1). Set in the worker process *before* jax is first imported, so it is
# applied via the worker-group env, not relied upon from the launcher.
DEFAULT_TRAINER_ENV: dict[str, str] = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}


class JaxPolicyConfig(TypedDict, total=False):
    """Trainer-side JAX backend configuration (``policy.jax_cfg``).

    Resolved from YAML/Hydra (an OmegaConf node) at runtime; ``total=False``
    because every key has a driver-side default and configs set only the subset
    they override. Mesh axis names reuse the torch logical names (dp/cp/tp/ep)
    to keep parity with the refit reshard logic.
    """

    enabled: bool                       # select the JAX trainer backend
    tensor_parallel_size: int           # tp mesh degree
    context_parallel_size: int          # cp mesh degree
    data_parallel_replicate_size: int   # dp replicate degree (dp_shard derived to fill world)
    expert_parallel_size: int           # ep degree (MoE; J8)
    moe_token_dispatcher: str           # "local" (EP=1) or "alltoall" (EP>1); MoE only
    moe_load_balance_coeff: float       # aux-loss-free bias step size (>0 enables); null disables
    param_dtype: str                    # parameter dtype, e.g. "bfloat16" (trainer stays bf16)
    env_vars: dict[str, str]            # per-worker env overrides (merged over DEFAULT_TRAINER_ENV)


def jax_backend_enabled(policy_config: Mapping[str, Any]) -> bool:
    """Return True when the policy config selects the JAX trainer backend.

    Reads ``policy.jax_cfg.enabled``. Absent/false leaves the torch DTensor
    backend as the default.
    """
    jax_cfg = policy_config.get("jax_cfg") or {}
    return bool(jax_cfg.get("enabled", False))


def resolve_trainer_env(jax_cfg: Mapping[str, Any]) -> dict[str, str]:
    """Compute the per-worker env for the JAX trainer worker group.

    Starts from ``DEFAULT_TRAINER_ENV`` and overlays any user-supplied
    ``jax_cfg.env_vars`` (user values win), so an operator can add or override
    XLA flags (e.g. ``XLA_PYTHON_CLIENT_MEM_FRACTION``) without losing the
    preallocation default. Returns a plain ``dict[str, str]`` suitable for
    ``RayWorkerGroup(env_vars=...)``.
    """
    env: dict[str, str] = dict(DEFAULT_TRAINER_ENV)
    user_env = jax_cfg.get("env_vars") or {}
    for key, value in user_env.items():
        env[str(key)] = str(value)
    return env
