from typing import Any
from datasets import load_dataset
from torch.utils.data import Dataset
from dockyard_rl.data import processors
from dockyard_rl.data.interfaces import TaskDataSpec

_CHOICE_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
]

class MMLUProDataset(Dataset):
    """MMLU-Pro evaluation dataset (10 answer choices per question).

    Args:
        task_data_spec: Task specification.
        tokenizer: HuggingFace tokenizer.
        max_input_seq_length: Maximum input length.
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

        raw = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
        self.data = [self._format(row) for row in raw]

    def _format(self, row: Any) -> dict[str, Any]:
        options = row["options"]
        labels = _CHOICE_LABELS[: len(options)]
        choice_str = "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, options))
        question = f"{row['question']}\n\n{choice_str}"
        answer_label = _CHOICE_LABELS[int(row["answer_index"])]
        return {
            "question": question,
            "answer": answer_label,
            "category": row.get("category", ""),
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.data[idx]
        datum = processors.text_preprocessor(
            datum_dict={"input": row["question"]},
            task_data_spec=self.task_data_spec,
            tokenizer=self.tokenizer,
            max_input_seq_length=self.max_input_seq_length,
        )
        datum["answer"] = row["answer"]
        datum["category"] = row["category"]
        return datum