"""On-policy distillation training launcher.

The student generates rollouts against environments; the frozen teacher scores
those rollouts by providing top-k logits; the student is trained to match the
teacher distribution via `DistillationLossFn`. This path therefore needs a
generation backend (student) and environments, like GRPO, plus a separate
`teacher_policy` block for the frozen teacher.

Usage:
    python3 examples/run_distillation.py \\
        --config examples/configs/distillation.yaml \\
        cluster.gpus_per_node=8 \\
        policy.model_name=Qwen/Qwen2.5-1.5B-Instruct \\
        teacher_policy.model_name=Qwen/Qwen2.5-7B-Instruct

CLI overrides follow Hydra dot-notation: key=value or key.nested=value.
"""

import argparse
import os
import pprint
from typing import Any, cast

from omegaconf import OmegaConf

from dockyard_rl.algorithms.distillation import (
    MasterConfig,
    distillation_train,
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


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run on-policy distillation.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config. Defaults to examples/configs/distillation.yaml.",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "distillation.yaml"
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
    print(f"📁 Log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(f"📁 Checkpoint directory: {config.checkpointing['checkpoint_dir']}")

    init_ray(log_dir=config.logger.get("log_dir"))

    tokenizer = get_tokenizer(config.policy["tokenizer"])

    # Finalise the student's generation config (pad token IDs, etc.).
    assert config.policy["generation"] is not None, (
        "A generation config is required for on-policy distillation."
    )
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    # Response data + environments (the 4-tuple form, like GRPO). `env` is an
    # extra field on the distillation MasterConfig (not a declared attribute),
    # so it is read from the model's extra fields.
    env_configs = cast("dict[str, Any]", (config.model_extra or {}).get("env"))
    train_dataset, val_dataset, task_to_env, val_task_to_env = cast(
        "tuple[Any, Any, dict[str, Any], dict[str, Any]]",
        setup_response_data(
            cast(Any, tokenizer),
            cast(Any, config.data),
            env_configs=env_configs,
        ),
    )

    (
        student_policy,
        teacher_policy,
        student_cluster,
        teacher_cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        distillation_save_state,
        master_config,
    ) = setup(
        config,
        cast(Any, tokenizer),
        train_dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    )

    print("🚀 Running on-policy distillation training")
    distillation_train(
        student_policy,
        teacher_policy,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        distillation_save_state,
        task_to_env,
        val_task_to_env,
    )

    print("All done.")


if __name__ == "__main__":
    main()
