"""GRPO training launcher for SWE-bench coding agent tasks.

Key characteristics:
- Uses SWEBenchDataset from dockyard_rl.data.datasets.swe_bench rather
  than the math dataset loader.
- Environment wiring uses dockyard_rl.environments.code_environment
  instead of the math environment.
- All imports use dockyard_rl.* packages.
- uv run removed — python3 directly as sys.executable is the interpreter.
- SGLang backend validation removed.

Usage:
    python3 examples/run_grpo_swe.py \\
        --config examples/configs/grpo_swe.yaml \\
        cluster.gpus_per_node=8 \\
        cluster.num_nodes=4 \\
        policy.model_name=Qwen/Qwen2.5-7B-Instruct

CLI overrides follow Hydra dot-notation: key=value or key.nested=value.
"""

import argparse
import os
import pprint
from typing import Any, cast

from omegaconf import OmegaConf

from dockyard_rl.algorithms.grpo import (
    MasterConfig,
    async_grpo_train,
    grpo_train,
    setup,
)
from dockyard_rl.algorithms.utils import get_tokenizer
from dockyard_rl.cluster.bootstrap import init_ray
from dockyard_rl.data.utils import setup_response_data
from dockyard_rl.models.generation import configure_generation_config
from dockyard_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from dockyard_rl.utils.logger import get_next_experiment_dir


# ── Argument parsing
def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run GRPO coding-agent training on SWE-bench."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.  Defaults to examples/configs/grpo_swe.yaml.",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


# ── Validation
def _validate_async_grpo_config(config: MasterConfig) -> None:
    """Raise NotImplementedError for features incompatible with async GRPO."""
    async_cfg = config.grpo.get("async_grpo", {})
    if not async_cfg.get("enabled", False):
        return

    # Dynamic sampling not supported with async GRPO.
    if config.grpo.get("use_dynamic_sampling", False):
        raise NotImplementedError(
            "use_dynamic_sampling is not supported with async GRPO."
        )

    # Reward scaling and shaping not supported with async GRPO.
    for feature in ("reward_scaling", "reward_shaping"):
        if feature in config.grpo and config.grpo[feature].get("enabled", False):
            raise NotImplementedError(
                f"{feature} is not supported with async GRPO."
            )

    # Multiple dataloaders not supported with async GRPO.
    if config.data.get("use_multiple_dataloader", False):
        raise NotImplementedError(
            "use_multiple_dataloader is not supported with async GRPO."
        )

    # Importance sampling correction required for async GRPO.
    if not config.loss_fn.use_importance_sampling_correction:
        raise ValueError(
            "loss_fn.use_importance_sampling_correction must be true "
            "when async_grpo.enabled=true."
        )


# ── Main
def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_swe.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"CLI overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config_dict = cast("dict[str, Any]", OmegaConf.to_container(config, resolve=True))
    config = MasterConfig(**config_dict)
    print("Applied CLI overrides.")

    print("Final config:")
    pprint.pprint(config.model_dump())

    # Auto-increment experiment directory so reruns don't clobber logs.
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📁 Log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(f"📁 Checkpoint directory: {config.checkpointing['checkpoint_dir']}")

    # Validate async GRPO constraints early to surface errors before
    # any expensive model loading.
    _validate_async_grpo_config(config)

    # Initialise Ray cluster.
    init_ray(log_dir=config.logger.get("log_dir"))

    # Tokenizer (or VLM processor). The computer-use / OSWorld path sets
    # policy.tokenizer.use_processor=true so the multimodal AutoProcessor is
    # loaded; the CUA rollout encodes per-turn screenshots through it.
    tokenizer = get_tokenizer(
        config.policy["tokenizer"],
        get_processor=bool(config.policy["tokenizer"].get("use_processor", False)),
    )
    assert config.policy["generation"] is not None, (
        "A generation config is required for GRPO."
    )

    # Generation config finalisation (pad token IDs, etc.).
    has_refit_draft_weights = bool(config.policy["draft"]["enabled"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
    )

    # Dataset and environment. The canonical path loads datasets via the
    # registry (binding each dataset's processor) and creates environments via
    # create_env, returning the task_name → environment maps GRPO consumes.
    dataset, val_dataset, task_to_env, val_task_to_env = cast(
        "tuple[Any, Any, dict[str, Any], dict[str, Any]]",
        setup_response_data(
            tokenizer   = cast(Any, tokenizer),
            data_config = cast(Any, config.data),
            env_configs = config.env,
            # Structured tool-use protocol (grpo.structured_tool_use): threaded to
            # the data layer so tool-use processors advertise the env's tool
            # registry in the prompt. None/disabled → fenced-text prompts unchanged.
            structured_tool_use = config.grpo.get("structured_tool_use"),
        ),
    )

    # Training setup.
    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
        weight_synchronizer,
    ) = setup(config, tokenizer, dataset, val_dataset)

    # Run training.
    async_cfg = config.grpo.get("async_grpo", {})
    if async_cfg.get("enabled", False):
        print("🚀 Running async GRPO training")
        async_grpo_train(
            policy                    = policy,
            policy_generation         = policy_generation,
            dataloader                = dataloader,
            val_dataloader            = val_dataloader,
            tokenizer                 = tokenizer,
            loss_fn                   = loss_fn,
            task_to_env               = task_to_env,
            val_task_to_env           = val_task_to_env,
            logger                    = logger,
            checkpointer              = checkpointer,
            grpo_save_state           = grpo_state,
            master_config             = master_config,
            weight_synchronizer       = weight_synchronizer,
            max_trajectory_age_steps  = async_cfg["max_trajectory_age_steps"],
        )
    else:
        print("🚀 Running synchronous GRPO training")
        grpo_train(
            policy            = policy,
            policy_generation = policy_generation,
            wrapped_dataloader = dataloader,
            val_dataloader    = val_dataloader,
            tokenizer         = tokenizer,
            loss_fn           = loss_fn,
            task_to_env       = task_to_env,
            val_task_to_env   = val_task_to_env,
            logger            = logger,
            checkpointer      = checkpointer,
            grpo_save_state   = grpo_state,
            master_config     = master_config,
            weight_synchronizer = weight_synchronizer,
        )

    print("All done.")

if __name__ == "__main__":
    main()