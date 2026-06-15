from typing import Any
from datasets import load_dataset
from dockyard_rl.data.datasets.raw_dataset import RawDataset

class SquadDataset(RawDataset):
    """Simple wrapper around the squad dataset.

    Args:
        split: Split name for the dataset, default is "train"
    """

    def __init__(self, split: str = "train", **kwargs) -> None:
        self.task_name = "squad"

        # load from huggingface
        self.dataset = load_dataset("rajpurkar/squad")[split]

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": data["context"],
                },
                {
                    "role": "user",
                    "content": data["question"],
                },
                {
                    "role": "assistant",
                    "content": data["answers"]["text"][0],
                },
            ],
            "task_name": self.task_name,
        }