from typing import Any, cast
from datasets import load_dataset
from torch.utils.data import Dataset
from dockyard_rl.data import processors
from dockyard_rl.data.interfaces import TaskDataSpec

_HF_ID_MAP = {
    "aime_2024": "HuggingFaceH4/aime_2024",
    "aime_2025": "HuggingFaceH4/aime_2025",
}

class AIMEDataset(Dataset):
    """AIME 2024 / 2025 evaluation dataset.

    Args:
        task_data_spec: Task specification.
        tokenizer: HuggingFace tokenizer.
        max_input_seq_length: Maximum input length.
        year: "aime_2024" or "aime_2025" (default: "aime_2024").
        split: Dataset split (default: "train" — AIME has no test split on HF).
    """

    def __init__(
        self,
        task_data_spec: TaskDataSpec,
        tokenizer,
        max_input_seq_length: int,
        year: str = "aime_2024",
        split: str = "train",
    ):
        if year not in _HF_ID_MAP:
            raise ValueError(
                f"Unknown AIME year: {year!r}. Choose from {sorted(_HF_ID_MAP)}."
            )
        self.task_data_spec = task_data_spec
        self.tokenizer = tokenizer
        self.max_input_seq_length = max_input_seq_length

        raw = load_dataset(_HF_ID_MAP[year], split=split)
        self.data = [self._format(cast(dict[str, Any], row)) for row in raw]

    def _format(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "problem": row["problem"],
            "answer": str(row["answer"]),
            "problem_id": row.get("id", ""),
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.data[idx]
        datum = processors.text_preprocessor(
            datum_dict={"input": row["problem"]},
            task_data_spec=self.task_data_spec,
            tokenizer=self.tokenizer,
            max_input_seq_length=self.max_input_seq_length,
        )
        datum["answer"] = row["answer"]
        datum["problem_id"] = row["problem_id"]
        return datum