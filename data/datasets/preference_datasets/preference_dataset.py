from typing import Any
from dockyard_rl.data.datasets.raw_dataset import RawDataset
from dockyard_rl.data.datasets.utils import load_dataset_from_path

class PreferenceDataset(RawDataset):
    """Generic preference dataset loader from a local path or HuggingFace Hub.

    Expects the dataset to have "chosen" and "rejected" columns, each
    containing an OpenAI-style message list.

    Args:
        data_path: Local file path (JSONL/CSV/Parquet) or HuggingFace Hub ID.
        split: Dataset split to load (default: "train").
        chosen_key: Column name for the chosen response (default: "chosen").
        rejected_key: Column name for the rejected response (default: "rejected").
        split_validation_size: Fraction to reserve for validation (default: 0.05).
        seed: Random seed for the train/val split (default: 42).
    """

    task_name = "preference"

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        split_validation_size: float = 0.05,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key

        self.dataset = load_dataset_from_path(data_path, data_split=split)

        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=[
                c
                for c in self.dataset.column_names
                if c not in {chosen_key, rejected_key}
            ],
        )

        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "chosen": data[self.chosen_key],
            "rejected": data[self.rejected_key],
            "task_name": self.task_name,
        }