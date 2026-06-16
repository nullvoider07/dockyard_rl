from typing import Any
from dockyard_rl.data.datasets.raw_dataset import RawDataset
from dockyard_rl.data.datasets.utils import load_dataset_from_path

class BinaryPreferenceDataset(RawDataset):
    """Preference dataset where each row has a context + two responses and a binary label.

    Expects columns: context (list of messages), response1, response2, label (0 or 1).
    Label 0 means response1 is preferred; label 1 means response2 is preferred.

    Args:
        data_path: Local file path or HuggingFace Hub ID.
        split: Dataset split (default: "train").
        context_key: Column for the shared context (default: "context").
        response1_key: Column for the first response (default: "response1").
        response2_key: Column for the second response (default: "response2").
        label_key: Column for the preference label (default: "label").
        split_validation_size: Fraction reserved for validation (default: 0.05).
        seed: Random seed (default: 42).
    """

    task_name = "binary_preference"

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        context_key: str = "context",
        response1_key: str = "response1",
        response2_key: str = "response2",
        label_key: str = "label",
        split_validation_size: float = 0.05,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.context_key = context_key
        self.response1_key = response1_key
        self.response2_key = response2_key
        self.label_key = label_key

        self.dataset = load_dataset_from_path(data_path, data_split=split)

        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        label = data[self.label_key]
        if label == 0:
            chosen = data[self.response1_key]
            rejected = data[self.response2_key]
        else:
            chosen = data[self.response2_key]
            rejected = data[self.response1_key]

        context = data[self.context_key]
        if isinstance(context, str):
            context = [{"role": "user", "content": context}]

        chosen_messages = context + (
            chosen
            if isinstance(chosen, list)
            else [{"role": "assistant", "content": chosen}]
        )
        rejected_messages = context + (
            rejected
            if isinstance(rejected, list)
            else [{"role": "assistant", "content": rejected}]
        )

        return {
            "chosen": chosen_messages,
            "rejected": rejected_messages,
            "task_name": self.task_name,
        }