"""Configuration loading and CLI-override utilities for Project Dockyard.

These helpers back the example launchers (e.g. ``examples/run_grpo_swe.py``)::

    register_omegaconf_resolvers()                 # once, before any resolution
    cfg = load_config(path)                        # DictConfig from a YAML file
    cfg = parse_hydra_overrides(cfg, cli_args)     # apply `key=value` overrides
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = MasterConfig(**cfg)

Resolvers
---------
``mul`` is the only custom OmegaConf resolver the shipped configs use — it
backs the token-budget arithmetic in ``policy.dynamic_batching`` /
``policy.sequence_packing``, e.g.::

    ${mul:${policy.max_total_sequence_length}, ${policy.train_micro_batch_size}}

It preserves ``int`` when both operands are integral so token-count keys do
not silently become floats.
"""

from __future__ import annotations

import os
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

_RESOLVERS_REGISTERED = False


def register_omegaconf_resolvers() -> None:
    """Register Dockyard's custom OmegaConf resolvers.

    Idempotent: safe to call more than once. Must be called before any
    config interpolation is resolved (i.e. before ``OmegaConf.to_container``
    with ``resolve=True``).
    """
    global _RESOLVERS_REGISTERED
    if _RESOLVERS_REGISTERED:
        return

    def _mul(a: Any, b: Any) -> Any:
        product = a * b
        if isinstance(a, int) and isinstance(b, int):
            return int(product)
        return product

    OmegaConf.register_new_resolver("mul", _mul, replace=True)
    _RESOLVERS_REGISTERED = True


def load_config(config_path: str) -> DictConfig:
    """Load a YAML config file into an OmegaConf ``DictConfig``.

    Interpolations are left unresolved; resolve them downstream with
    ``OmegaConf.to_container(cfg, resolve=True)`` after registering resolvers.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        The loaded config as a ``DictConfig``.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        TypeError:         If the top-level YAML node is not a mapping.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(
            "Expected the top-level config to be a mapping, got "
            f"{type(cfg).__name__} from {config_path!r}."
        )
    return cfg


def _delete_dotted_key(cfg: DictConfig, dotted_key: str) -> None:
    """Delete ``a.b.c`` from ``cfg`` (used for Hydra ``~key`` overrides)."""
    parts = dotted_key.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        node = node[part]
    with open_dict(node):
        del node[parts[-1]]


def parse_hydra_overrides(cfg: DictConfig, overrides: list[str]) -> DictConfig:
    """Apply Hydra-style dot-notation CLI overrides to a config.

    Supports the three Hydra override forms:

        key.subkey=value   change (or add) a value
        +key=value         add a new key
        ~key               delete a key

    Values are parsed with Hydra's grammar, so ``null``/``true``/``[1,2]``/
    numbers are converted to their native types rather than left as strings.

    Args:
        cfg:       Config to override. A copy is returned; ``cfg`` is untouched.
        overrides: Override strings, e.g. ``["cluster.num_nodes=4"]``.

    Returns:
        A new ``DictConfig`` with the overrides applied.
    """
    merged: DictConfig = cfg.copy()
    if not overrides:
        return merged

    from hydra.core.override_parser.overrides_parser import OverridesParser

    parser = OverridesParser.create()
    parsed = parser.parse_overrides(overrides=overrides)

    with open_dict(merged):
        for override in parsed:
            key = override.key_or_group
            if override.is_delete():
                _delete_dotted_key(merged, key)
            else:
                OmegaConf.update(
                    merged, key, override.value(), merge=True, force_add=True
                )
    return merged
