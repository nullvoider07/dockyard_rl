from typing import Any
from torch.utils.data import Dataset
from dockyard_rl.data.interfaces import TaskDataSpec

# Lazy imports — classes are only imported when needed so that optional
# heavy dependencies (soundfile, scipy, etc.) don't block startup
EVAL_DATASET_REGISTRY: dict[str, str] = {
    "math": "dockyard_rl.data.datasets.eval_datasets.math.MathDataset",
    "aime": "dockyard_rl.data.datasets.eval_datasets.aime.AIMEDataset",
    "aime_2024": "dockyard_rl.data.datasets.eval_datasets.aime.AIMEDataset",
    "aime_2025": "dockyard_rl.data.datasets.eval_datasets.aime.AIMEDataset",
    "gpqa": "dockyard_rl.data.datasets.eval_datasets.gpqa.GPQADataset",
    "gpqa_diamond": "dockyard_rl.data.datasets.eval_datasets.gpqa.GPQADataset",
    "mmlu": "dockyard_rl.data.datasets.eval_datasets.mmlu.MMLUDataset",
    "mmlu_pro": "dockyard_rl.data.datasets.eval_datasets.mmlu_pro.MMLUProDataset",
    "mmau": "dockyard_rl.data.datasets.eval_datasets.mmau.MMAUDataset",
    "local_math": "dockyard_rl.data.datasets.eval_datasets.local_math_dataset.LocalMathDataset",
}

def load_eval_dataset(
    eval_cfg: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: Any,
    max_input_seq_length: int,
) -> Dataset:
    """Instantiate an eval dataset from a config dict.

    Args:
        eval_cfg: Must contain a "type" key matching EVAL_DATASET_REGISTRY.
                  All other keys are forwarded as constructor kwargs.
        task_data_spec: Task specification with prompt templates / system prompts.
        tokenizer: HuggingFace tokenizer.
        max_input_seq_length: Maximum token length for inputs.

    Returns:
        A torch Dataset instance.
    """
    import importlib

    dataset_type = eval_cfg.get("type")
    if dataset_type is None:
        raise ValueError("eval_cfg must contain a 'type' key.")

    if dataset_type not in EVAL_DATASET_REGISTRY:
        raise ValueError(
            f"Unknown eval dataset type: {dataset_type!r}. "
            f"Available: {sorted(EVAL_DATASET_REGISTRY)}."
        )

    fqn = EVAL_DATASET_REGISTRY[dataset_type]
    module_path, cls_name = fqn.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)

    kwargs = {k: v for k, v in eval_cfg.items() if k != "type"}

    # For AIME shorthand keys, forward the year automatically.
    if dataset_type in {"aime_2024", "aime_2025"} and "year" not in kwargs:
        kwargs["year"] = dataset_type

    # For GPQA shorthand, forward the subset automatically.
    if dataset_type == "gpqa_diamond" and "subset" not in kwargs:
        kwargs["subset"] = "gpqa_diamond"

    return cls(
        task_data_spec=task_data_spec,
        tokenizer=tokenizer,
        max_input_seq_length=max_input_seq_length,
        **kwargs,
    )

__all__ = [
    "EVAL_DATASET_REGISTRY",
    "load_eval_dataset",
]