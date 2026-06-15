"""Online / iterative DPO launcher.

Closes the DPO loop: generate K candidates per prompt from the current policy,
score them with a judge, build best-vs-worst chosen/rejected pairs (margin-gated),
and run a DPO step — then repeat. The CPU-testable core lives in
`algorithms/online_dpo.py` (judge, pair construction, loop); this launcher binds
its injected callables (`generate_fn` / `train_fn`) to the live generation and
trainer fleets.

Fleet wiring is reused from GRPO: `grpo.setup` builds the policy, generation
engine, cluster, and weight synchronizer; the DPO loss is built by
`build_preference_loss` and passed per `policy.train` call. The `generate_fn`
(generate + decode) and `train_fn` (reference logprobs + train + weight sync)
bodies follow the documented generation/policy APIs but execute only on live GPU
fleets (hardware-deferred — see handoff/hardware-deferred-validation.md).

Usage:
    python3 examples/run_online_dpo.py \\
        --config examples/configs/online_dpo.yaml \\
        policy.model_name=Qwen/Qwen2.5-7B-Instruct cluster.gpus_per_node=8

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
from dockyard_rl.algorithms.online_dpo import (
    Candidate,
    OnlineDPOConfig,
    PreferenceJudge,
    online_dpo_train,
)
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
    parser = argparse.ArgumentParser(description="Run online (iterative) DPO.")
    parser.add_argument("--config", type=str, default=None)
    args, overrides = parser.parse_known_args()
    return args, overrides


def _extract_prompt_token_ids(batch: Any) -> list[torch.Tensor]:
    """Pull per-sample prompt token-id tensors from a dataloader batch.

    The bring-up integration point: a processed sample carries its prompt under
    the message log; this returns one 1-D LongTensor per prompt for generation.
    """
    message_logs = batch["message_log"]
    prompts: list[torch.Tensor] = []
    for log in message_logs:
        # The prompt is everything up to (not including) the assistant turn.
        ids = [
            turn["token_ids"]
            for turn in log
            if turn.get("role") != "assistant" and "token_ids" in turn
        ]
        prompts.append(torch.cat(ids) if ids else torch.empty(0, dtype=torch.long))
    return prompts


def _make_generate_fn(policy_generation, tokenizer, k: int, max_len: int):
    """Build generate_fn: K candidates per prompt via the generation engine."""
    pad_id = tokenizer.pad_token_id

    def generate_fn(prompts: list[Any]):
        # Repeat each prompt K times so one generate() call yields K samples each.
        repeated: list[torch.Tensor] = []
        for p in prompts:
            repeated.extend([p] * k)
        input_lengths = torch.tensor([t.shape[0] for t in repeated], dtype=torch.long)
        width = int(input_lengths.max().item()) if len(repeated) else 0
        input_ids = torch.full((len(repeated), width), pad_id, dtype=torch.long)
        for i, t in enumerate(repeated):
            input_ids[i, : t.shape[0]] = t

        gen_in = BatchedDataDict({"input_ids": input_ids, "input_lengths": input_lengths})
        out = policy_generation.generate(gen_in, greedy=False)
        output_ids = out["output_ids"]
        total_lengths = out["unpadded_sequence_lengths"]

        candidates: list[Candidate] = []
        for i in range(len(repeated)):
            in_len = int(input_lengths[i].item())
            total = int(total_lengths[i].item())
            resp_ids = output_ids[i, in_len:total]
            prompt_ids = repeated[i]
            candidates.append(
                Candidate(
                    prompt_token_ids=prompt_ids,
                    response_token_ids=resp_ids,
                    prompt_text=tokenizer.decode(prompt_ids, skip_special_tokens=True),
                    response_text=tokenizer.decode(resp_ids, skip_special_tokens=True),
                )
            )
        # Regroup into K-per-prompt candidate groups.
        return [candidates[j * k : (j + 1) * k] for j in range(len(prompts))]

    return generate_fn


def _make_train_fn(policy, policy_generation, weight_synchronizer, loss_fn, mbs: int):
    """Build train_fn: reference logprobs → DPO step → weight sync."""

    def train_fn(batch: Any) -> dict[str, Any]:
        policy.prepare_for_training()
        ref = policy.get_reference_policy_logprobs(batch, micro_batch_size=mbs)[
            "reference_logprobs"
        ]
        batch["reference_policy_logprobs"] = torch.roll(ref, -1, dims=-1)
        results = policy.train(batch, loss_fn, gbs=batch.size, mbs=mbs)
        policy.finish_training()
        # Push fresh weights to the generation engine for the next iteration.
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
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "online_dpo.yaml"
        )

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
    assert config.policy["generation"] is not None, "online DPO requires a generation config."
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    dataset, val_dataset, _task_to_env, _val_task_to_env = cast(
        "tuple[Any, Any, dict[str, Any], dict[str, Any]]",
        setup_response_data(
            tokenizer=cast(Any, tokenizer),
            data_config=cast(Any, config.data),
            env_configs=config.env,
        ),
    )

    # Reuse GRPO setup for the fleet (policy, generation, cluster, weight sync).
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

    # DPO loss (built from the `dpo` config block; passed per policy.train call).
    dpo_cfg = cast("dict[str, Any]", config.model_extra or {}).get("dpo", {})
    online_cfg = cast(OnlineDPOConfig, cast("dict[str, Any]", config.model_extra or {})["online_dpo"])
    loss_fn = build_preference_loss(dpo_cfg.get("loss_variant"), cast(Any, dpo_cfg))

    # Preference judge: verifiable reward preferred where applicable, else the LLM judge.
    judge_client = HLEJudgeClient(
        base_url=online_cfg.get("judge_base_url"),
        api_key=online_cfg.get("judge_api_key"),
        model=online_cfg.get("judge_model"),
        timeout=online_cfg.get("judge_timeout"),
    )
    judge = PreferenceJudge(
        judge=judge_client,
        prefer_verifiable=online_cfg["prefer_verifiable"],
        temperature=online_cfg.get("judge_temperature", 0.0),
    )

    mbs = int(config.policy["train_micro_batch_size"])
    generate_fn = _make_generate_fn(
        policy_generation, tokenizer, online_cfg["candidates_per_prompt"],
        int(config.policy["max_total_sequence_length"]),
    )
    train_fn = _make_train_fn(policy, policy_generation, weight_synchronizer, loss_fn, mbs)
    build_batch_fn = partial(
        preference_collate_fn,
        tokenizer=tokenizer,
        make_sequence_length_divisible_by=int(
            config.policy["make_sequence_length_divisible_by"]
        ),
        add_loss_mask=True,
    )

    # Prompt batches: one generation step per dataloader batch.
    prompt_batches = [_extract_prompt_token_ids(b) for b in dataloader]

    # Prime the generation engine with the initial weights.
    weight_synchronizer.mark_stale()
    if weight_synchronizer.is_stale:
        weight_synchronizer.sync_weights()
        policy_generation.prepare_for_generation()

    print("🚀 Running online DPO training")
    online_dpo_train(
        prompt_batches,
        generate_fn=generate_fn,
        judge=judge,
        build_batch_fn=build_batch_fn,
        train_fn=train_fn,
        config=online_cfg,
        logger=logger,
    )
    print("All done.")


if __name__ == "__main__":
    main()
