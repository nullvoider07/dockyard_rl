from typing import Any
from datasets import load_dataset
from dockyard_rl.data.datasets.raw_dataset import RawDataset

class HelpSteer3PreferenceDataset(RawDataset):
    """HelpSteer3 dataset formatted as chosen/rejected preference pairs.

    Pairs with overall_preference == 0 are excluded (ties carry no signal).

    Args:
        split: Dataset split (default: "train").
        split_validation_size: Fraction reserved for validation (default: 0.05).
        seed: Random seed (default: 42).
    """

    task_name = "helpsteer3_preference"

    def __init__(
        self,
        split: str = "train",
        split_validation_size: float = 0.05,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.dataset = load_dataset("nvidia/HelpSteer3", "preference")[split]

        # Drop ties before mapping — they have no preference signal.
        self.dataset = self.dataset.filter(
            lambda x: x["overall_preference"] != 0
        )

        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        context = data["context"]
        if isinstance(context, str):
            context = [{"role": "user", "content": context}]

        if data["overall_preference"] < 0:
            chosen_text = data["response1"]
            rejected_text = data["response2"]
        else:
            chosen_text = data["response2"]
            rejected_text = data["response1"]

        chosen = context + [{"role": "assistant", "content": chosen_text}]
        rejected = context + [{"role": "assistant", "content": rejected_text}]

        return {
            "chosen": chosen,
            "rejected": rejected,
            "task_name": self.task_name,
        }