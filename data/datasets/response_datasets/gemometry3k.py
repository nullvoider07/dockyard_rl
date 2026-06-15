from typing import Any
from datasets import load_dataset
from dockyard_rl.data.datasets.raw_dataset import RawDataset
from dockyard_rl.data.datasets.utils import pil_to_base64

def format_geometry3k_dataset(
    example: dict[str, Any], return_pil: bool = False
) -> dict[str, Any]:
    """Format the Geometry3K dataset into an OpenAI-API-like message log."""
    # isolate single image
    if isinstance(example["images"], list):
        example["image"] = example["images"][0]

    user_content = [
        {
            "type": "image",
            "image": pil_to_base64(example["image"])
            if not return_pil
            else example["image"],
        },
        {
            "type": "text",
            "text": str(example["problem"]).replace("<image>", ""),
        },
    ]

    assistant_content = str(example["answer"])

    ret = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "task_name": example["task_name"],
    }
    return ret

class Geometry3KDataset(RawDataset):
    """Simple wrapper around the Geometry3K dataset.

    Args:
        split: Split name for the dataset, default is "train"
    """

    def __init__(self, split: str = "train", **kwargs):
        # train, validation, and test are supported splits.
        assert split in ["train", "validation", "test"], (
            f"Invalid split: {split}. Please use 'train' or 'validation' or 'test'."
        )

        self.task_name = "geometry3k"

        # this dataset will process the image during training using `format_geometry3k_dataset`
        self.dataset = load_dataset("hiyouga/geometry3k")[split]

        # format - disable features to avoid schema conflicts
        self.dataset = self.dataset.add_column(
            "task_name", [self.task_name] * len(self.dataset)
        )