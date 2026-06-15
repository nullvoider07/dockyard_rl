from typing import Any
from datasets import load_dataset
from torch.utils.data import Dataset
from dockyard_rl.data import processors
from dockyard_rl.data.interfaces import TaskDataSpec

class MathDataset(Dataset):
    """MATH-500 evaluation dataset.

    Args:
        task_data_spec: Task specification including system prompt and prompt template.
        tokenizer: HuggingFace tokenizer.
        max_input_seq_length: Maximum input sequence length.
        split: Dataset split (default: "test").
    """

    def __init__(
        self,
        task_data_spec: TaskDataSpec,
        tokenizer,
        max_input_seq_length: int,
        split: str = "test",
    ):
        self.task_data_spec = task_data_spec
        self.tokenizer = tokenizer
        self.max_input_seq_length = max_input_seq_length

        raw = load_dataset("HuggingFaceH4/MATH-500", split=split)
        self.data = [self._format(row) for row in raw]

    def _format(self, row: Any) -> dict[str, Any]:
        return {
            "problem": row["problem"],
            "answer": row["answer"],
            "subject": row.get("subject", ""),
            "level": row.get("level", ""),
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
        datum["subject"] = row["subject"]
        datum["level"] = row["level"]
        return datum