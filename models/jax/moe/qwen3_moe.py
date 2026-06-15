"""Qwen3-MoE model in Flax NNX (J8d).

Reuses the dense Qwen3 attention / norm / embedding / RoPE / causal-mask from
``models/jax/models/qwen3.py`` and replaces each routed-MoE layer's MLP with the
native ``MoEBlock`` (router + grouped-GEMM experts). Verified against
``transformers.models.qwen3_moe`` (5.8.1):

  - a layer is sparse iff ``layer_idx not in mlp_only_layers`` and
    ``num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0`` — else a
    dense ``Qwen3MLP`` (HF ``Qwen3MoeMLP``) with ``intermediate_size``;
  - the router is softmax top-k with ``norm_topk_prob`` (= ``route_norm``);
  - the routing weight scales the expert OUTPUT (``score_before_experts=False``),
    so ``MoEBlock`` matches HF's ``current_hidden_states * top_k_weights``;
  - HF stores fused experts ``gate_up_proj (E,2F,D)`` (gate||up) + ``down_proj
    (E,D,F)``; the loader slices them into ``w1/w3/w2_EFD`` (no transpose, the
    layout already matches the native experts).

Qwen3-MoE has no shared expert. ``ragged_dot`` lowers on CPU, so a tiny model is
forward-logit parity-checked against HF on CPU (``test_jax_qwen3_moe_parity.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from dockyard_rl.models.jax.layers import Embedding, Linear, RMSNorm
from dockyard_rl.models.jax.models.qwen3 import (
    Qwen3Attention,
    Qwen3Config,
    Qwen3MLP,
    _causal_bias,
    _rope_cos_sin,
)
from dockyard_rl.models.jax.models.qwen3 import _COLUMN_PARALLEL
from dockyard_rl.models.jax.moe.alltoall import make_token_dispatcher
from dockyard_rl.models.jax.moe.block import MoEBlock
from dockyard_rl.models.jax.moe.experts import GroupedExperts
from dockyard_rl.models.jax.moe.router import TokenChoiceTopKRouter

Array = jax.Array


@dataclass(frozen=True)
class Qwen3MoeConfig:
    """Qwen3-MoE hyperparameters (dense Qwen3 fields + the routed-MoE fields)."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int  # dense-MLP FFN dim (non-sparse layers)
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    # routed-MoE
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    norm_topk_prob: bool = False

    @property
    def dense_cfg(self) -> Qwen3Config:
        """Dense Qwen3 view for the shared attention / dense-MLP modules."""
        return Qwen3Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    def is_sparse_layer(self, layer_idx: int) -> bool:
        return (layer_idx not in self.mlp_only_layers) and (
            self.num_experts > 0 and (layer_idx + 1) % self.decoder_sparse_step == 0
        )

    @classmethod
    def from_hf_config(cls, hf: Any) -> "Qwen3MoeConfig":
        def get(name: str, default: Any = None) -> Any:
            if isinstance(hf, dict):
                return hf.get(name, default)
            return getattr(hf, name, default)

        head_dim = get("head_dim") or (get("hidden_size") // get("num_attention_heads"))
        rope_params = get("rope_parameters") or {}
        rope_theta = rope_params.get("rope_theta") if isinstance(rope_params, dict) else None
        if rope_theta is None:
            rope_theta = get("rope_theta", 10000.0)
        rope_type = rope_params.get("rope_type", "default") if isinstance(rope_params, dict) else "default"
        if rope_type != "default":
            raise NotImplementedError(
                f"Qwen3-MoE JAX supports only default RoPE; got rope_type={rope_type!r}."
            )
        return cls(
            vocab_size=int(get("vocab_size")),
            hidden_size=int(get("hidden_size")),
            intermediate_size=int(get("intermediate_size")),
            num_hidden_layers=int(get("num_hidden_layers")),
            num_attention_heads=int(get("num_attention_heads")),
            num_key_value_heads=int(get("num_key_value_heads")),
            head_dim=int(head_dim),
            rms_norm_eps=float(get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope_theta),
            tie_word_embeddings=bool(get("tie_word_embeddings", False)),
            num_experts=int(get("num_experts")),
            num_experts_per_tok=int(get("num_experts_per_tok")),
            moe_intermediate_size=int(get("moe_intermediate_size")),
            decoder_sparse_step=int(get("decoder_sparse_step", 1)),
            mlp_only_layers=tuple(get("mlp_only_layers", ()) or ()),
            norm_topk_prob=bool(get("norm_topk_prob", False)),
        )


def build_moe_block(
    cfg: Qwen3MoeConfig, *, rngs: nnx.Rngs, param_dtype: jnp.dtype,
    load_balance_coeff: Optional[float], token_dispatcher: str = "local",
) -> MoEBlock:
    """Native ``MoEBlock`` for a Qwen3-MoE sparse layer (HF-faithful settings).

    ``token_dispatcher``: ``local`` (EP=1, CPU path) or ``alltoall`` (EP>1; its
    ``ep`` axis is wired at parallelize time on a live mesh — HV).
    """
    router = TokenChoiceTopKRouter(
        cfg.hidden_size,
        cfg.num_experts,
        top_k=cfg.num_experts_per_tok,
        score_func="softmax",
        route_norm=cfg.norm_topk_prob,
        route_scale=1.0,
        rngs=rngs,
        param_dtype=param_dtype,
    )
    # HF applies the routing weight to the expert OUTPUT -> score_before_experts=False.
    dispatcher = make_token_dispatcher(
        token_dispatcher, cfg.num_experts, cfg.num_experts_per_tok, score_before_experts=False
    )
    experts = GroupedExperts(
        cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts, dispatcher,
        rngs=rngs, param_dtype=param_dtype,
    )
    return MoEBlock(
        router, experts, num_experts=cfg.num_experts, shared_experts=None,
        load_balance_coeff=load_balance_coeff,
    )


class Qwen3MoeDecoderLayer(nnx.Module):
    def __init__(
        self, cfg: Qwen3MoeConfig, layer_idx: int, *, rngs: nnx.Rngs,
        param_dtype: jnp.dtype, load_balance_coeff: Optional[float],
        token_dispatcher: str = "local",
    ) -> None:
        dcfg = cfg.dense_cfg
        self.self_attn = Qwen3Attention(dcfg, rngs=rngs, param_dtype=param_dtype)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, param_dtype=param_dtype)
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, param_dtype=param_dtype
        )
        if cfg.is_sparse_layer(layer_idx):
            self.mlp: nnx.Module = build_moe_block(
                cfg, rngs=rngs, param_dtype=param_dtype, load_balance_coeff=load_balance_coeff,
                token_dispatcher=token_dispatcher,
            )
        else:
            self.mlp = Qwen3MLP(dcfg, rngs=rngs, param_dtype=param_dtype)

    def __call__(self, x: Array, cos: Array, sin: Array, attn_bias: Array) -> Array:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attn_bias)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3MoeModel(nnx.Module):
    def __init__(
        self, cfg: Qwen3MoeConfig, *, rngs: nnx.Rngs, param_dtype: jnp.dtype = jnp.float32,
        load_balance_coeff: Optional[float] = None, token_dispatcher: str = "local",
    ) -> None:
        self.cfg = cfg
        self.embed_tokens = Embedding(cfg.vocab_size, cfg.hidden_size, rngs=rngs, param_dtype=param_dtype)
        self.layers = nnx.List(
            [
                Qwen3MoeDecoderLayer(
                    cfg, i, rngs=rngs, param_dtype=param_dtype, load_balance_coeff=load_balance_coeff,
                    token_dispatcher=token_dispatcher,
                )
                for i in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, param_dtype=param_dtype)

    def __call__(self, input_ids: Array, position_ids: Optional[Array] = None) -> Array:
        b, s = input_ids.shape
        if position_ids is None:
            position_ids = jnp.arange(s, dtype=jnp.int32)[None, :].repeat(b, axis=0)
        x = self.embed_tokens(input_ids)
        cos, sin = _rope_cos_sin(position_ids, self.cfg.head_dim, self.cfg.rope_theta)
        cos = cos.astype(x.dtype)
        sin = sin.astype(x.dtype)
        attn_bias = _causal_bias(s, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, attn_bias)
        return self.norm(x)


class Qwen3MoeForCausalLM(nnx.Module):
    """Qwen3-MoE LM head over the base model (tied / untied like dense Qwen3)."""

    def __init__(
        self, cfg: Qwen3MoeConfig, *, rngs: nnx.Rngs, param_dtype: jnp.dtype = jnp.float32,
        load_balance_coeff: Optional[float] = None, token_dispatcher: str = "local",
    ) -> None:
        self.cfg = cfg
        self.tied = cfg.tie_word_embeddings
        self.model = Qwen3MoeModel(
            cfg, rngs=rngs, param_dtype=param_dtype, load_balance_coeff=load_balance_coeff,
            token_dispatcher=token_dispatcher,
        )
        self.lm_head = (
            None
            if self.tied
            else Linear(
                cfg.hidden_size, cfg.vocab_size, rngs=rngs, param_dtype=param_dtype,
                sharding=_COLUMN_PARALLEL,
            )
        )

    def __call__(self, input_ids: Array, position_ids: Optional[Array] = None) -> Array:
        hidden = self.model(input_ids, position_ids)
        if self.tied:
            embedding = self.model.embed_tokens.embedding[...]
            return jnp.einsum("...h,vh->...v", hidden, embedding)
        assert self.lm_head is not None
        return self.lm_head(hidden)
