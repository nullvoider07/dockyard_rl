from typing import Any
from datasets import load_dataset
from torch.utils.data import Dataset
from dockyard_rl.data import processors
from dockyard_rl.data.interfaces import TaskDataSpec

_CHOICE_LABELS = ["A", "B", "C", "D"]

class MMLUDataset(Dataset):
    """MMLU (Massive Multitask Language Understanding) evaluation dataset.

    Args:
        task_data_spec: Task specification.
        tokenizer: HuggingFace tokenizer.
        max_input_seq_length: Maximum input length.
        subject: MMLU subject name, or "all" to load all subjects (default: "all").
        split: Dataset split (default: "test").
        multilingual: If True, loads the multilingual MMLU variant (default: False).
        language: Language code for multilingual MMLU (default: "en").
    """

    def __init__(
        self,
        task_data_spec: TaskDataSpec,
        tokenizer,
        max_input_seq_length: int,
        subject: str = "all",
        split: str = "test",
        multilingual: bool = False,
        language: str = "en",
    ):
        self.task_data_spec = task_data_spec
        self.tokenizer = tokenizer
        self.max_input_seq_length = max_input_seq_length

        if multilingual:
            raw = load_dataset("alexandrainst/m_mmlu", language, split=split)
        elif subject == "all":
            raw = load_dataset("cais/mmlu", "all", split=split)
        else:
            raw = load_dataset("cais/mmlu", subject, split=split)

        self.data = [self._format(row) for row in raw]

    def _format(self, row: Any) -> dict[str, Any]:
        choices = row["choices"]
        choice_str = "\n".join(
            f"{lbl}. {txt}" for lbl, txt in zip(_CHOICE_LABELS, choices)
        )
        question = f"{row['question']}\n\n{choice_str}"
        answer_label = _CHOICE_LABELS[int(row["answer"])]
        return {
            "question": question,
            "answer": answer_label,
            "subject": row.get("subject", ""),
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
        datum["subject"] = row["subject"]
        return datum