from typing import Any
from datasets import load_dataset
from dockyard_rl.data.datasets.raw_dataset import RawDataset

class Tulu3PreferenceDataset(RawDataset):
    """Tulu3 preference dataset formatted as chosen/rejected pairs.

    Args:
        split: Dataset split (default: "train").
        split_validation_size: Fraction reserved for validation (default: 0.05).
        seed: Random seed (default: 42).
        max_samples: Optional cap on dataset size.
    """

    task_name = "tulu3_preference"

    def __init__(
        self,
        split: str = "train",
        split_validation_size: float = 0.05,
        seed: int = 42,
        max_samples: int | None = None,
        **kwargs,
    ) -> None:
        self.dataset = load_dataset("allenai/tulu-3-pref-data-reformat")[split]

        if max_samples is not None and max_samples > 0:
            self.dataset = self.dataset.shuffle(seed=seed).select(
                range(min(max_samples, len(self.dataset)))
            )

        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "chosen": data["chosen"],
            "rejected": data["rejected"],
            "task_name": self.task_name,
        }