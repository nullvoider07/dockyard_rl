from typing import Any
from datasets import load_dataset
from dockyard_rl.data.datasets.raw_dataset import RawDataset

class AIME2024Dataset(RawDataset):
    """Simple wrapper around the AIME2024 dataset with train split.

    Args:
        repeat: Number of times to repeat the dataset, default is 16
    """

    def __init__(self, repeat: int = 16, **kwargs) -> None:
        self.task_name = "AIME2024"

        # load from huggingface
        self.dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        # repeat the dataset
        self.dataset = self.dataset.repeat(repeat)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": data["problem"]},
                {"role": "assistant", "content": data["answer"]},
            ],
            "task_name": self.task_name,
        }