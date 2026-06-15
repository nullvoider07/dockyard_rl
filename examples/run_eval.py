"""Standalone evaluation launcher.

Runs generation on a held-out set against a verifier environment and reports the
reward metrics, reusing the GRPO setup + `validate()` path (no optimizer step).
The model is loaded into the generation engine via the normal refit handshake,
then `grpo.validate` scores `grpo.max_val_samples` prompts.

Two data modes (selected by the config):
  * **Eval-dataset mode** — when an `eval` block with a `dataset` is present, the
    benchmark is loaded from EVAL_DATASET_REGISTRY (math / aime / gpqa / mmlu /
    mmlu_pro / mmau / local_math) via `load_eval_dataset`, the ground-truth answer
    and task name are threaded onto each sample, and a verifier environment
    (`eval.env_name`, e.g. `math`) scores the generations.
  * **Response/env mode** — otherwise, a response dataset + environment is built
    with `setup_response_data` (same path GRPO training uses for validation).

Usage:
    python3 examples/run_eval.py \\
        --config examples/configs/eval.yaml \\
        policy.model_name=Qwen/Qwen2.5-7B-Instruct cluster.gpus_per_node=8

CLI overrides follow Hydra dot-notation: key=value or key.nested=value.
"""

import argparse
import os
import pprint
from typing import Any, cast

from omegaconf import OmegaConf
from torch.utils.data import Dataset

from dockyard_rl.algorithms.grpo import MasterConfig, setup, validate
from dockyard_rl.algorithms.utils import get_tokenizer
from dockyard_rl.cluster.bootstrap import init_ray
from dockyard_rl.data.datasets.eval_datasets import load_eval_dataset
from dockyard_rl.data.interfaces import TaskDataSpec
from dockyard_rl.data.utils import setup_response_data
from dockyard_rl.environments.utils import create_env
from dockyard_rl.models.generation import configure_generation_config
from dockyard_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from dockyard_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run standalone evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config. Defaults to examples/configs/eval.yaml.",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


class _EvalSampleBridge(Dataset):
    """Wrap an eval dataset so each sample is rollout/verifier-ready.

    Sets ``task_name`` (so the rollout routes the sample to the verifier env) and
    promotes the dataset's top-level ground-truth ``answer`` into
    ``extra_env_info["ground_truth"]`` (where the env reads it), unless already set.
    """

    def __init__(self, base: Any, task_name: str) -> None:
        self._base = base
        self._task = task_name

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        datum = dict(self._base[idx])
        datum.setdefault("task_name", self._task)
        env_info = dict(datum.get("extra_env_info") or {})
        if not env_info.get("ground_truth") and "answer" in datum:
            env_info["ground_truth"] = str(datum["answer"])
        datum["extra_env_info"] = env_info
        datum.setdefault("idx", idx)
        return datum


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "eval.yaml")

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
    print(f"📁 Log directory: {config.logger['log_dir']}")

    init_ray(log_dir=config.logger.get("log_dir"))

    tokenizer = get_tokenizer(
        config.policy["tokenizer"],
        get_processor=bool(config.policy["tokenizer"].get("use_processor", False)),
    )
    assert config.policy["generation"] is not None, (
        "A generation config is required for evaluation."
    )
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    eval_cfg = cast("dict[str, Any]", config.model_extra or {}).get("eval")
    if eval_cfg and eval_cfg.get("dataset"):
        # ── Eval-dataset mode (EVAL_DATASET_REGISTRY benchmarks) ──────────────
        task_name = eval_cfg.get("task_name") or eval_cfg["dataset"]["type"]
        task_spec = TaskDataSpec(
            task_name=task_name,
            prompt_file=eval_cfg.get("prompt_file"),
            system_prompt_file=eval_cfg.get("system_prompt_file"),
        )
        raw_eval = load_eval_dataset(
            cast("dict[str, Any]", eval_cfg["dataset"]),
            task_spec,
            cast(Any, tokenizer),
            int(config.data["max_input_seq_length"]),
        )
        eval_dataset: Any = _EvalSampleBridge(raw_eval, task_name)
        env_name = eval_cfg["env_name"]
        val_task_to_env = {task_name: create_env(env_name, config.env[env_name])}
        # grpo.setup builds the policy + generation engine + val_dataloader from
        # the eval dataset (passed as both train and validation).
        (
            _policy,
            policy_generation,
            _cluster,
            _dataloader,
            val_dataloader,
            _loss_fn,
            logger,
            _checkpointer,
            _grpo_state,
            master_config,
            weight_synchronizer,
        ) = setup(config, tokenizer, eval_dataset, eval_dataset)
    else:
        # ── Response/env mode (same path GRPO uses for validation) ────────────
        _dataset, _val_dataset, _t2e, val_task_to_env = cast(
            "tuple[Any, Any, dict[str, Any], dict[str, Any]]",
            setup_response_data(
                tokenizer=cast(Any, tokenizer),
                data_config=cast(Any, config.data),
                env_configs=config.env,
            ),
        )
        (
            _policy,
            policy_generation,
            _cluster,
            _dataloader,
            val_dataloader,
            _loss_fn,
            logger,
            _checkpointer,
            _grpo_state,
            master_config,
            weight_synchronizer,
        ) = setup(config, tokenizer, _dataset, _val_dataset)

    # Load the policy weights into the generation engine before scoring.
    weight_synchronizer.mark_stale()
    if weight_synchronizer.is_stale:
        weight_synchronizer.sync_weights()
        policy_generation.prepare_for_generation()

    print("🚀 Running evaluation")
    val_metrics, _ = validate(
        policy_generation,
        val_dataloader,
        tokenizer,
        val_task_to_env,
        step=0,
        master_config=master_config,
        logger=logger,
    )

    print("Evaluation metrics:")
    pprint.pprint(val_metrics)
    policy_generation.finish_generation()
    print("All done.")


if __name__ == "__main__":
    main()
