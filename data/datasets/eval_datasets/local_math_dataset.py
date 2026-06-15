from typing import Any
from torch.utils.data import Dataset
from dockyard_rl.data import processors
from dockyard_rl.data.datasets.utils import load_dataset_from_path
from dockyard_rl.data.interfaces import TaskDataSpec

class LocalMathDataset(Dataset):
    """Evaluation dataset loaded from a local file (JSONL, CSV, or Parquet).

    Expected columns: "problem" (or "question") and "answer".
    An optional "subject" column is preserved if present.

    Args:
        task_data_spec: Task specification.
        tokenizer: HuggingFace tokenizer.
        max_input_seq_length: Maximum input length.
        data_path: Path to the local data file.
        problem_key: Column name for the problem/question (default: "problem").
        answer_key: Column name for the answer (default: "answer").
    """

    def __init__(
        self,
        task_data_spec: TaskDataSpec,
        tokenizer,
        max_input_seq_length: int,
        data_path: str,
        problem_key: str = "problem",
        answer_key: str = "answer",
    ):
        self.task_data_spec = task_data_spec
        self.tokenizer = tokenizer
        self.max_input_seq_length = max_input_seq_length
        self.problem_key = problem_key
        self.answer_key = answer_key

        raw = load_dataset_from_path(data_path)
        self.data = [self._format(row) for row in raw]

    def _format(self, row: Any) -> dict[str, Any]:
        problem = row.get(self.problem_key) or row.get("question", "")
        answer = str(row.get(self.answer_key, ""))
        return {
            "problem": problem,
            "answer": answer,
            "subject": row.get("subject", ""),
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
        return datum