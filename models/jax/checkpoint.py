"""Orbax checkpointing + HF-safetensors export for the JAX worker.

Clean-break Orbax checkpoints (decision: new lineage; cross-backend *weight*
warm-start is via the HF-safetensors export + the J1 loader, not a torch-DCP
bridge — that optimizer-state bridge is the J7b stretch). A checkpoint is a
pytree of pure-dict NNX states: the model params, optionally the optimizer
state (optax moments + the wrapped model), and the step counter.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from dockyard_rl.models.jax.refit import iter_refit_state_dict


def save_checkpoint(
    path: str, model: nnx.Module, *, optimizer: Optional[nnx.Optimizer] = None, step: int = 0
) -> None:
    """Save model params (+ optional optimizer state) + step to ``path`` (Orbax)."""
    ckpt: dict[str, Any] = {
        "model": nnx.to_pure_dict(nnx.state(model)),
        "step": jnp.asarray(int(step), dtype=jnp.int32),
    }
    if optimizer is not None:
        ckpt["optimizer"] = nnx.to_pure_dict(nnx.state(optimizer))
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(os.path.abspath(path), ckpt)
    ckptr.wait_until_finished()


def restore_checkpoint(
    path: str, model: nnx.Module, *, optimizer: Optional[nnx.Optimizer] = None
) -> int:
    """Restore params (+ optional optimizer state) into ``model``; return the step.

    ``model`` (and ``optimizer``) must already be constructed with the same
    config — they supply the abstract structure Orbax restores into.
    """
    target: dict[str, Any] = {
        "model": nnx.to_pure_dict(nnx.state(model)),
        "step": jnp.asarray(0, dtype=jnp.int32),
    }
    if optimizer is not None:
        target["optimizer"] = nnx.to_pure_dict(nnx.state(optimizer))

    ckptr = ocp.StandardCheckpointer()
    restored = ckptr.restore(os.path.abspath(path), target=target)

    mstate = nnx.state(model)
    nnx.replace_by_pure_dict(mstate, restored["model"])
    nnx.update(model, mstate)
    if optimizer is not None:
        ostate = nnx.state(optimizer)
        nnx.replace_by_pure_dict(ostate, restored["optimizer"])
        nnx.update(optimizer, ostate)
    return int(restored["step"])


def export_hf_safetensors(model: nnx.Module, path: str) -> None:
    """Export the model's params as an HF-named safetensors file.

    Reuses the J5 refit name-map (`iter_refit_state_dict`), so the file is
    loadable by the J1 HF loader (`weights.load_hf_state_dict`) and by vLLM /
    transformers — the cross-backend weight-interchange path.
    """
    from safetensors.torch import save_file

    state_dict = {name: tensor.contiguous() for name, tensor in iter_refit_state_dict(model)}
    save_file(state_dict, os.path.abspath(path))
