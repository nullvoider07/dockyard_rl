"""Recipe + infra config loader for ``dockyard-k8s``.

Given a recipe path, loads and merges four layers in priority order (low to
high):

  1. Shipped defaults        — ``dockyard_k8s/defaults/defaults.example.yaml``
  2. User defaults           — ``~/.config/dockyard-k8s/defaults.yaml`` (optional)
  3. Recipe-level ``infra:`` — the ``infra:`` key on the recipe, or a ``--infra`` file
  4. CLI Hydra overrides     — ``infra.scheduler.queue=my-queue``, etc.

Layers 1-3 are YAML; layer 4 is a list of Hydra-style strings. The merged
``infra`` mapping is validated through :class:`dockyard_k8s.schema.InfraConfig`.

The rest of the recipe (``policy`` / ``grpo`` / ``data`` / ``env`` / ...) is
loaded via ``dockyard_rl.utils.config`` when importable, so defaults
inheritance (``defaults: ../grpo_swe.yaml``) and ``${mul:...}`` resolvers are
handled exactly as the training entrypoints handle them. When ``dockyard_rl``
is not importable (e.g. the CLI is installed standalone on an ops host), a
minimal OmegaConf fallback handles a single recipe with ``defaults:``
inheritance.
"""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .schema import InfraConfig


def get_username() -> str:
    """Return the local OS username, sanitised for K8s resource names."""
    raw = os.environ.get("DOCKYARD_K8S_USER") or getpass.getuser()
    return raw.lower().replace("_", "-").replace(".", "-")


def _register_resolvers() -> None:
    if not OmegaConf.has_resolver("user"):
        OmegaConf.register_new_resolver("user", get_username)
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
    if not OmegaConf.has_resolver("div"):
        OmegaConf.register_new_resolver("div", lambda a, b: a / b)
    if not OmegaConf.has_resolver("max"):
        OmegaConf.register_new_resolver("max", lambda a, b: max(a, b))


_SHIPPED_DEFAULTS = Path(__file__).parent / "defaults" / "defaults.example.yaml"
_USER_DEFAULTS = Path(
    os.environ.get(
        "DOCKYARD_K8S_DEFAULTS",
        Path.home() / ".config" / "dockyard-k8s" / "defaults.yaml",
    )
)


@dataclass
class LoadedConfig:
    """Bundle returned by :func:`load_recipe_with_infra`.

    ``recipe`` holds the resolved recipe with ``infra`` removed (so it can be
    passed to the dockyard_rl entrypoint as-is). ``infra`` is the validated
    :class:`InfraConfig`. ``source_path`` is the recipe path that was loaded.
    """

    recipe: DictConfig
    infra: InfraConfig
    source_path: Path


def load_recipe_with_infra(
    recipe_path: str | Path,
    overrides: list[str] | None = None,
    *,
    infra_path: str | Path | None = None,
) -> LoadedConfig:
    """Load a dockyard_rl recipe plus its infra config and return both.

    Two supported layouts:

    * Split — pass ``infra_path`` to point at a dedicated infra YAML. The
      recipe must not also declare an ``infra:`` key.
    * Bundled — the recipe carries an ``infra:`` top-level key; ``infra_path``
      is ``None``.

    ``infra.*`` overrides apply to the infra layer; all others apply to the recipe.
    """
    overrides = overrides or []
    recipe_path = Path(recipe_path).resolve()
    _register_resolvers()

    recipe_overrides, infra_overrides = _partition_overrides(overrides)
    recipe = _load_recipe(recipe_path, overrides=recipe_overrides)

    infra_raw = _merge_infra(
        recipe,
        infra_path=Path(infra_path).resolve() if infra_path else None,
        overrides=infra_overrides,
    )
    recipe.pop("infra", None)  # peel any recipe-level infra: off

    infra_container = OmegaConf.to_container(infra_raw, resolve=True)
    if not isinstance(infra_container, dict):
        raise RuntimeError("internal: infra config did not resolve to a mapping")
    if not infra_container.get("namespace"):
        infra_container["namespace"] = _infer_kube_namespace()
    infra = InfraConfig.model_validate(infra_container)

    return LoadedConfig(recipe=recipe, infra=infra, source_path=recipe_path)


_SA_NS_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")


def _infer_kube_namespace() -> str:
    """Default ``infra.namespace`` to the active kube context's namespace.

    Tries the pod service-account file, then the current kubeconfig context,
    and finally ``default``.
    """
    try:
        if _SA_NS_PATH.exists():
            ns = _SA_NS_PATH.read_text().strip()
            if ns:
                return ns
    except OSError:
        pass
    try:
        from kubernetes import config as k8s_config  # type: ignore[import-not-found]

        _, active = k8s_config.list_kube_config_contexts()
        ns = ((active or {}).get("context") or {}).get("namespace")
        if ns:
            return ns
    except Exception:
        pass
    return "default"


def _partition_overrides(overrides: list[str]) -> tuple[list[str], list[str]]:
    """Split Hydra overrides: ``infra.*`` to the infra layer, rest to recipe.

    Keeps the recipe loader from seeing (and rejecting) infra.* keys on
    strict dockyard_rl configs that use struct mode.
    """
    recipe_side: list[str] = []
    infra_side: list[str] = []
    for o in overrides:
        body = o.lstrip("+~")
        if body.startswith("infra.") or body == "infra":
            leading = o[: len(o) - len(body)]
            infra_side.append(leading + body[len("infra.") :])
        else:
            recipe_side.append(o)
    return recipe_side, infra_side


def _load_recipe(recipe_path: Path, overrides: list[str]) -> DictConfig:
    """Load a recipe YAML via dockyard_rl's loader, else an OmegaConf fallback."""
    try:
        from dockyard_rl.utils.config import (  # type: ignore[import-not-found]
            load_config,
            parse_hydra_overrides,
            register_omegaconf_resolvers,
        )
    except ImportError:
        return _load_recipe_fallback(recipe_path, overrides)

    register_omegaconf_resolvers()
    cfg = load_config(str(recipe_path))
    if overrides:
        cfg = parse_hydra_overrides(cfg, overrides)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"recipe at {recipe_path} did not load as a mapping")
    return cfg


def _load_recipe_fallback(recipe_path: Path, overrides: list[str]) -> DictConfig:
    """OmegaConf-only loader used when dockyard_rl is unavailable."""
    cfg = _load_with_inheritance(recipe_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"recipe at {recipe_path} did not load as a mapping")
    return cfg


def _load_with_inheritance(path: Path) -> DictConfig:
    """Walk a recipe's ``defaults:`` chain and return the merged DictConfig."""
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"{path} did not load as a mapping")

    if "defaults" in cfg:
        raw = cfg.pop("defaults")
        defaults: list[str] = (
            [str(raw)] if isinstance(raw, (str, Path)) else [str(x) for x in raw]
        )
        base: DictConfig = OmegaConf.create({})
        for rel in defaults:
            parent = (path.parent / rel).resolve()
            parent_cfg = _load_with_inheritance(parent)
            merged = OmegaConf.merge(base, parent_cfg)
            if not isinstance(merged, DictConfig):
                raise ValueError(f"defaults merge for {path} produced non-mapping")
            base = merged
        merged = OmegaConf.merge(base, cfg)
        if not isinstance(merged, DictConfig):
            raise ValueError(f"inheritance merge for {path} produced non-mapping")
        cfg = merged
    return cfg


def _merge_infra(
    recipe: DictConfig,
    *,
    infra_path: Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Stack shipped defaults < user defaults < (infra file | recipe.infra) < CLI."""
    shipped = _load_yaml_if_present(_SHIPPED_DEFAULTS, required=True)
    assert shipped is not None  # required=True raises rather than returning None
    user = _load_yaml_if_present(_USER_DEFAULTS, required=False)

    infra_layer: DictConfig
    if infra_path is not None:
        if "infra" in recipe:
            raise ValueError(
                "infra config supplied via --infra but the recipe also contains "
                "an `infra:` key — choose one or the other."
            )
        infra_layer = _pick_infra(_load_with_inheritance(infra_path))
    else:
        infra_layer = _extract_recipe_infra(recipe)

    # Strip anchor-only scratch sections (`_shared: &foo ...`) — they carry no
    # infra meaning but survive YAML parsing.
    for k in list(infra_layer.keys()):
        if isinstance(k, str) and k.startswith("_"):
            infra_layer.pop(k)

    merged = OmegaConf.merge(
        _pick_infra(shipped),
        _pick_infra(user) if user is not None else OmegaConf.create({}),
        infra_layer,
    )
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    if not isinstance(merged, DictConfig):
        raise RuntimeError("internal: infra merge did not produce a DictConfig")
    return merged


def _pick_infra(cfg: DictConfig) -> DictConfig:
    """A defaults file may be either ``{infra: {...}}`` or just the infra body."""
    if "infra" in cfg:
        inner = cfg["infra"]
        if not isinstance(inner, DictConfig):
            raise ValueError("defaults file has non-mapping `infra:` key")
        return _detach(inner)
    return cfg


def _extract_recipe_infra(recipe: DictConfig) -> DictConfig:
    if "infra" not in recipe:
        return OmegaConf.create({})
    inner = recipe["infra"]
    if not isinstance(inner, DictConfig):
        raise ValueError("recipe `infra:` key must be a mapping")
    return _detach(inner)


def _detach(cfg: DictConfig) -> DictConfig:
    """Return a parent-free copy so interpolations resolve from this root."""
    detached = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if not isinstance(detached, DictConfig):
        raise ValueError("internal: detached config is not a mapping")
    return detached


def _load_yaml_if_present(path: Path, *, required: bool) -> DictConfig | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"shipped defaults missing: {path}")
        return None
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise ValueError(f"{path} did not load as a mapping")
    return loaded


__all__ = ["LoadedConfig", "get_username", "load_recipe_with_infra"]
