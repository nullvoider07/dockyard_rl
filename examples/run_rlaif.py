"""RLAIF / Constitutional-AI launcher.

Generates preference pairs without human labels and trains on them with the DPO
step: for each prompt the current policy produces a response, a constitutional
critique→revise pass produces an improved response (chosen) vs. the original
(rejected), and a DPO step optimizes toward the revision. Reuses the tested
RLAIF pair constructors (`algorithms/rlaif.py::build_rlaif_preference_data`) and
the GRPO fleet setup; the generation + train binding executes on live GPU fleets
(hardware-deferred).

Usage:
    python3 examples/run_rlaif.py \\
        --config examples/configs/rlaif.yaml \\
        policy.model_name=Qwen/Qwen2.5-7B-Instruct cluster.gpus_per_node=8 \\
        rlaif.llm_base_url=... rlaif.llm_model=...

CLI overrides follow Hydra dot-notation: key=value or key.nested=value.
"""

import argparse
import os
import pprint
from functools import partial
from typing import Any, cast

import torch
from omegaconf import OmegaConf

from dockyard_rl.algorithms.grpo import MasterConfig, setup
from dockyard_rl.algorithms.loss import build_preference_loss
from dockyard_rl.algorithms.rlaif import build_rlaif_preference_data
from dockyard_rl.algorithms.utils import get_tokenizer
from dockyard_rl.cluster.bootstrap import init_ray
from dockyard_rl.data.collate_fn import preference_collate_fn
from dockyard_rl.data.utils import setup_response_data
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.models.generation import configure_generation_config
from dockyard_rl.rewards.hle_grader import HLEJudgeClient
from dockyard_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from dockyard_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run RLAIF / Constitutional-AI training.")
    parser.add_argument("--config", type=str, default=None)
    args, overrides = parser.parse_known_args()
    return args, overrides


def _prompt_token_ids(batch: Any) -> list[torch.Tensor]:
    """One prompt token-id tensor per sample (everything before the assistant turn)."""
    prompts: list[torch.Tensor] = []
    for log in batch["message_log"]:
        ids = [t["token_ids"] for t in log if t.get("role") != "assistant" and "token_ids" in t]
        prompts.append(torch.cat(ids) if ids else torch.empty(0, dtype=torch.long))
    return prompts


def _generate_responses(policy_generation, tokenizer, prompts: list[torch.Tensor]):
    """One sampled response per prompt → (prompt_text, response_text) pairs."""
    pad_id = tokenizer.pad_token_id
    input_lengths = torch.tensor([t.shape[0] for t in prompts], dtype=torch.long)
    width = int(input_lengths.max().item()) if prompts else 0
    input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    for i, t in enumerate(prompts):
        input_ids[i, : t.shape[0]] = t
    gen_in = BatchedDataDict({"input_ids": input_ids, "input_lengths": input_lengths})
    out = policy_generation.generate(gen_in, greedy=False)
    output_ids = out["output_ids"]
    total_lengths = out["unpadded_sequence_lengths"]
    pairs: list[tuple[str, str]] = []
    for i in range(len(prompts)):
        in_len = int(input_lengths[i].item())
        total = int(total_lengths[i].item())
        prompt_text = tokenizer.decode(prompts[i], skip_special_tokens=True)
        response_text = tokenizer.decode(output_ids[i, in_len:total], skip_special_tokens=True)
        pairs.append((prompt_text, response_text))
    return pairs


def _make_train_fn(policy, policy_generation, weight_synchronizer, loss_fn, mbs: int):
    def train_fn(batch: Any) -> dict[str, Any]:
        policy.prepare_for_training()
        ref = policy.get_reference_policy_logprobs(batch, micro_batch_size=mbs)[
            "reference_logprobs"
        ]
        batch["reference_policy_logprobs"] = torch.roll(ref, -1, dims=-1)
        results = policy.train(batch, loss_fn, gbs=batch.size, mbs=mbs)
        policy.finish_training()
        weight_synchronizer.mark_stale()
        if weight_synchronizer.is_stale:
            weight_synchronizer.sync_weights()
            policy_generation.prepare_for_generation()
        return cast("dict[str, Any]", results.get("all_mb_metrics", results) or {})

    return train_fn


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "rlaif.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"CLI overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config_dict = cast("dict[str, Any]", OmegaConf.to_container(config, resolve=True))
    config = MasterConfig(**config_dict)
    print("Final config:")
    pprint.pprint(config.model_dump())

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    init_ray(log_dir=config.logger.get("log_dir"))

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, "RLAIF requires a generation config."
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    dataset, val_dataset, _t2e, _vt2e = cast(
        "tuple[Any, Any, dict[str, Any], dict[str, Any]]",
        setup_response_data(
            tokenizer=cast(Any, tokenizer),
            data_config=cast(Any, config.data),
            env_configs=config.env,
        ),
    )

    (
        policy,
        policy_generation,
        _cluster,
        dataloader,
        _val_dataloader,
        _grpo_loss_fn,
        logger,
        _checkpointer,
        _grpo_state,
        master_config,
        weight_synchronizer,
    ) = setup(config, tokenizer, dataset, val_dataset)

    extra = cast("dict[str, Any]", config.model_extra or {})
    rlaif_cfg = cast("dict[str, Any]", extra["rlaif"])
    dpo_cfg = cast("dict[str, Any]", extra.get("dpo", {}))
    loss_fn = build_preference_loss(dpo_cfg.get("loss_variant"), cast(Any, dpo_cfg))

    llm = HLEJudgeClient(
        base_url=rlaif_cfg.get("llm_base_url"),
        api_key=rlaif_cfg.get("llm_api_key"),
        model=rlaif_cfg.get("llm_model"),
        timeout=rlaif_cfg.get("llm_timeout"),
    )

    mbs = int(config.policy["train_micro_batch_size"])
    train_fn = _make_train_fn(policy, policy_generation, weight_synchronizer, loss_fn, mbs)
    collate = partial(
        preference_collate_fn,
        tokenizer=tokenizer,
        make_sequence_length_divisible_by=int(config.policy["make_sequence_length_divisible_by"]),
        add_loss_mask=True,
    )
    temperature = float(rlaif_cfg.get("temperature", 0.0))
    max_principles = rlaif_cfg.get("max_principles")
    max_steps = int(rlaif_cfg["max_num_steps"])

    weight_synchronizer.mark_stale()
    if weight_synchronizer.is_stale:
        weight_synchronizer.sync_weights()
        policy_generation.prepare_for_generation()

    print("🚀 Running RLAIF / Constitutional-AI training")
    consumed = 0
    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break
        prompts = _prompt_token_ids(batch)
        pr_pairs = _generate_responses(policy_generation, tokenizer, prompts)
        pref_data, gen_metrics = build_rlaif_preference_data(
            pr_pairs, cast("Any", llm), tokenizer,
            max_principles=max_principles, temperature=temperature, start_idx=consumed,
        )
        metrics: dict[str, Any] = dict(gen_metrics)
        if pref_data:
            collated = collate(pref_data)
            train_metrics = train_fn(collated)
            metrics.update({f"train/{k}": v for k, v in train_metrics.items()})
            metrics["trained"] = True
            consumed += len(pref_data)
        else:
            metrics["trained"] = False
        if logger is not None:
            logger.log_metrics(metrics, step, prefix="rlaif")

    print("All done.")


if __name__ == "__main__":
    main()
