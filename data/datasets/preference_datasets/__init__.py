from dockyard_rl.data.datasets.preference_datasets.binary_preference_dataset import (
    BinaryPreferenceDataset,
)
from dockyard_rl.data.datasets.preference_datasets.helpsteer3 import (
    HelpSteer3PreferenceDataset,
)
from dockyard_rl.data.datasets.preference_datasets.preference_dataset import (
    PreferenceDataset,
)
from dockyard_rl.data.datasets.preference_datasets.tulu3 import (
    Tulu3PreferenceDataset,
)

# Registry: maps dataset name → fully-qualified actor/class name.
# Used by setup_preference_data() in dockyard_rl.data.utils.
PREFERENCE_DATASET_REGISTRY: dict[str, str] = {
    "preference": "dockyard_rl.data.datasets.preference_datasets.preference_dataset.PreferenceDataset",
    "binary_preference": "dockyard_rl.data.datasets.preference_datasets.binary_preference_dataset.BinaryPreferenceDataset",
    "helpsteer3": "dockyard_rl.data.datasets.preference_datasets.helpsteer3.HelpSteer3PreferenceDataset",
    "tulu3": "dockyard_rl.data.datasets.preference_datasets.tulu3.Tulu3PreferenceDataset",
}

def load_preference_dataset(dataset_cfg: PreferenceDataset | BinaryPreferenceDataset) -> "RawDataset":  # type: ignore[name-defined]  # noqa: F821
    """Instantiate a preference dataset from a config dict.

    Args:
        dataset_cfg: Must contain a "type" key matching a PREFERENCE_DATASET_REGISTRY entry.
                     All other keys are forwarded as constructor kwargs.

    Returns:
        An instantiated RawDataset subclass.
    """
    import importlib

    dataset_type = dataset_cfg.get("type")
    if dataset_type is None:
        raise ValueError("dataset_cfg must contain a 'type' key.")

    if dataset_type not in PREFERENCE_DATASET_REGISTRY:
        raise ValueError(
            f"Unknown preference dataset type: {dataset_type!r}. "
            f"Available: {sorted(PREFERENCE_DATASET_REGISTRY)}"
        )

    fqn = PREFERENCE_DATASET_REGISTRY[dataset_type]
    module_path, cls_name = fqn.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)

    kwargs = {k: v for k, v in dataset_cfg.items() if k != "type"}
    return cls(**kwargs)

__all__ = [
    "BinaryPreferenceDataset",
    "HelpSteer3PreferenceDataset",
    "PreferenceDataset",
    "Tulu3PreferenceDataset",
    "PREFERENCE_DATASET_REGISTRY",
    "load_preference_dataset",
]