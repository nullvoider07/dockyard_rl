"""JAX trainer backend for Project Dockyard.

Re-platforms the trainer fleet from torch (DTensor/FSDP2 + autograd +
``torch.optim``) onto JAX (Flax NNX + ``jax.sharding`` + optax), below the
``models/policy/interfaces.py`` ABCs. Selected by ``policy.jax_cfg.enabled``,
mirroring the torch ``dtensor_cfg._v2`` flag; the torch worker stays the
default. Inference (vLLM/SGLang) and the sandbox fleet remain torch.

Module layout (filled in across phases J1–J11; see
``handoff/jax-trainer-replatform-plan.md``):

    config.py        JaxPolicyConfig schema + trainer-fleet env defaults   (J0)
    models/          Flax NNX model definitions (Qwen3 dense first)          (J1)
    weights.py       HF safetensors <-> NNX param loader + name map          (J1)
    sharding.py      jax.sharding.Mesh / PartitionSpec for dp/cp/tp          (J2)
    loss.py          pure-JAX loss + vocab-parallel logprob                  (J3)
    policy_worker.py JaxPolicyWorker (PolicyInterface / Colocatable...)      (J4)
    refit.py         JAX params -> dlpack -> torch, HF/vLLM name map         (J5)
    checkpoint.py    Orbax save/load                                          (J7)

Nothing here imports torch at module import time except at the explicit
torch<->jnp conversion boundaries (Seam 1/Seam 2).
"""

from dockyard_rl.models.jax.config import (
    DEFAULT_TRAINER_ENV,
    JaxPolicyConfig,
    jax_backend_enabled,
    resolve_trainer_env,
)

__all__ = [
    "DEFAULT_TRAINER_ENV",
    "JaxPolicyConfig",
    "jax_backend_enabled",
    "resolve_trainer_env",
]
