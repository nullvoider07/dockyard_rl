import random
from typing import Any
from datasets import load_dataset
from torch.utils.data import Dataset
from dockyard_rl.data import processors
from dockyard_rl.data.interfaces import TaskDataSpec

_SUBSET_MAP = {
    "gpqa_main": "gpqa_main",
    "gpqa_diamond": "gpqa_diamond",
    "gpqa_extended": "gpqa_extended",
}

_CHOICE_KEYS = ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
_CHOICE_LABELS = ["A", "B", "C", "D"]

class GPQADataset(Dataset):
    """GPQA (Graduate-level Problem Questions and Answers) evaluation dataset.

    Shuffles the answer choices per sample so position bias is minimised.

    Args:
        task_data_spec: Task specification.
        tokenizer: HuggingFace tokenizer.
        max_input_seq_length: Maximum input length.
        subset: One of "gpqa_main", "gpqa_diamond", "gpqa_extended" (default: "gpqa_diamond").
        split: Dataset split (default: "train").
        seed: Shuffle seed for answer positions (default: 42).
    """

    def __init__(
        self,
        task_data_spec: TaskDataSpec,
        tokenizer,
        max_input_seq_length: int,
        subset: str = "gpqa_diamond",
        split: str = "train",
        seed: int = 42,
    ):
        if subset not in _SUBSET_MAP:
            raise ValueError(
                f"Unknown GPQA subset: {subset!r}. Choose from {sorted(_SUBSET_MAP)}."
            )
        self.task_data_spec = task_data_spec
        self.tokenizer = tokenizer
        self.max_input_seq_length = max_input_seq_length
        self.rng = random.Random(seed)

        raw = load_dataset("Idavidrein/gpqa", _SUBSET_MAP[subset], split=split)
        self.data = [self._format(row) for row in raw]

    def _format(self, row: Any) -> dict[str, Any]:
        choices = [row[k] for k in _CHOICE_KEYS]
        correct = choices[0]  # "Correct Answer" is always index 0 in the raw data

        # Shuffle choices so correct answer lands at a random position
        shuffled = choices[:]
        self.rng.shuffle(shuffled)
        correct_label = _CHOICE_LABELS[shuffled.index(correct)]

        choice_str = "\n".join(
            f"{lbl}. {txt}" for lbl, txt in zip(_CHOICE_LABELS, shuffled)
        )
        question = f"{row['Question']}\n\n{choice_str}"

        return {
            "question": question,
            "answer": correct_label,
            "explanation": row.get("Explanation", ""),
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
        datum["explanation"] = row["explanation"]
        return datum