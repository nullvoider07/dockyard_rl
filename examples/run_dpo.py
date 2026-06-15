"""DPO training launcher.

Direct Preference Optimization on a (chosen, rejected) preference dataset. The
`DPOLossFn` (built inside `dpo.setup` from the `dpo` config block) supports the
DPO / DPOP / cDPO / R-DPO / IPO / KTO variants and the reference-free SimPO/ORPO
paths; the default is vanilla DPO. There is no generation in the offline path.

Note: DPO is incompatible with dynamic batching and sequence packing (both
reorder samples within a batch, breaking the chosen/rejected pairing); the
shipped config disables them.

Usage:
    python3 examples/run_dpo.py \\
        --config examples/configs/dpo.yaml \\
        cluster.gpus_per_node=8 \\
        policy.model_name=Qwen/Qwen2.5-1.5B-Instruct

CLI overrides follow Hydra dot-notation: key=value or key.nested=value.
"""

import argparse
import os
import pprint
from typing import Any, cast

from omegaconf import OmegaConf

from dockyard_rl.algorithms.dpo import MasterConfig, dpo_train, setup
from dockyard_rl.algorithms.utils import get_tokenizer
from dockyard_rl.cluster.bootstrap import init_ray
from dockyard_rl.data.utils import setup_preference_data
from dockyard_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from dockyard_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run DPO training.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. Defaults to examples/configs/dpo.yaml.",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "dpo.yaml")

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

    # Preference (chosen/rejected) data; validation is a dict of named sets.
    train_dataset, val_dataset = setup_preference_data(
        cast(Any, tokenizer), cast(Any, config.data)
    )

    (
        policy,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        dpo_save_state,
        master_config,
    ) = setup(config, cast(Any, tokenizer), train_dataset, val_dataset)

    print("🚀 Running DPO training")
    dpo_train(
        policy,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        dpo_save_state,
    )

    print("All done.")


if __name__ == "__main__":
    main()
