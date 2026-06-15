from typing import Any
from datasets import load_dataset
from dockyard_rl.data.datasets.raw_dataset import RawDataset

class DeepScalerDataset(RawDataset):
    """Simple wrapper around the DeepScaler dataset with train split."""

    def __init__(self, **kwargs) -> None:
        self.task_name = "DeepScaler"

        # load from huggingface
        self.dataset = load_dataset(
            "agentica-org/DeepScaleR-Preview-Dataset", split="train"
        )

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": data["problem"]},
                {"role": "assistant", "content": data["answer"]},
            ],
            "task_name": self.task_name,
        }