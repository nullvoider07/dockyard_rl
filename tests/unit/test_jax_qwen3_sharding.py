"""J2: sharded-vs-single-device parity for the JAX Qwen3 dense model.

XLA_FLAGS must request the host device count *before* jax initializes, so it is
set at module import. On a 4-device CPU mesh (dp=2, tp=2) the GSPMD-partitioned
forward must produce logits identical to the single-device (replicated) forward
of the same parameters. This exercises Megatron column/row tensor-parallelism
on the attention/MLP linears plus the per-head reshape under a sharded head dim.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402
nnx = pytest.importorskip("flax.nnx")

from dockyard_rl.models.jax.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from dockyard_rl.models.jax.sharding import build_mesh, data_sharding, shard_model


def _cfg() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,    # divisible by tp=2
        num_key_value_heads=2,    # divisible by tp=2
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )


@pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 host devices (set XLA_FLAGS first)")
def test_sharded_vs_single_device_parity() -> None:
    cfg = _cfg()
    model = Qwen3ForCausalLM(cfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)

    rng = np.random.default_rng(0)
    ids = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(4, 9)).astype(np.int64))

    # Reference: single-device replicated forward (params still on default device).
    ref = np.asarray(model(ids))

    mesh = build_mesh(dp=2, tp=2)
    shard_model(model, mesh)

    @nnx.jit
    def forward(m: Qwen3ForCausalLM, x: jax.Array) -> jax.Array:
        return m(x)

    sharded_ids = jax.device_put(ids, data_sharding(mesh))
    out = forward(model, sharded_ids)

    # Output logits should be tensor-parallel over the vocab dim (lm_head column-parallel).
    assert "tp" in str(out.sharding.spec)
    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 host devices")
def test_param_partition_specs() -> None:
    from dockyard_rl.models.jax.sharding import param_named_shardings

    cfg = _cfg()
    model = Qwen3ForCausalLM(cfg, rngs=nnx.Rngs(params=0), param_dtype=jnp.float32)
    mesh = build_mesh(dp=2, tp=2)
    specs = param_named_shardings(model, mesh)

    def spec_for(suffix: tuple) -> str:
        for path, ns in specs.items():
            if tuple(path)[-len(suffix):] == suffix:
                return str(ns.spec)
        raise KeyError(suffix)

    # Column-parallel: output dim sharded; row-parallel: input dim sharded.
    assert spec_for(("self_attn", "q_proj", "kernel")) == str(jax.sharding.PartitionSpec(None, "tp"))
    assert spec_for(("self_attn", "o_proj", "kernel")) == str(jax.sharding.PartitionSpec("tp", None))
    assert spec_for(("mlp", "down_proj", "kernel")) == str(jax.sharding.PartitionSpec("tp", None))
    assert spec_for(("lm_head", "kernel")) == str(jax.sharding.PartitionSpec(None, "tp"))
    # Embedding + norms replicated.
    assert spec_for(("embed_tokens", "embedding")) == str(jax.sharding.PartitionSpec())
