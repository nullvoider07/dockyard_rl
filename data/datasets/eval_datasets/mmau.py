import io
from typing import Any
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from torch.utils.data import Dataset
from dockyard_rl.data.datasets.response_datasets.avqa import _resample_audio
from dockyard_rl.data.interfaces import TaskDataSpec
from dockyard_rl.data.processors import vlm_hf_data_processor

_CHOICE_LABELS = ["A", "B", "C", "D"]

_PROMPT_TEMPLATE = (
    "{question}\n"
    "A. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}\n"
    "Answer with the option letter only."
)

class MMAUDataset(Dataset):
    """MMAU (Massive Multitask Audio Understanding) evaluation dataset.

    Loads audio + multiple-choice questions and returns samples suitable
    for audio-capable models (e.g. Qwen2.5-Omni).

    Args:
        task_data_spec: Task specification.
        tokenizer: HuggingFace tokenizer.
        max_input_seq_length: Maximum input length.
        split: "mini" or "full" (default: "mini").
        target_sr: Target audio sample rate in Hz (default: 16000).
    """

    def __init__(
        self,
        task_data_spec: TaskDataSpec,
        tokenizer,
        max_input_seq_length: int,
        split: str = "mini",
        target_sr: int = 16000,
    ):
        self.task_data_spec = task_data_spec
        self.tokenizer = tokenizer
        self.max_input_seq_length = max_input_seq_length
        self.target_sr = target_sr

        raw = load_dataset("lewtun/mmau", split=split)
        raw = raw.cast_column("audio", Audio(decode=False))
        self.data = [self._format(row) for row in raw]

    def _format(self, row: Any) -> dict[str, Any]:
        audio_raw = row["audio"]
        audio_array, orig_sr = sf.read(io.BytesIO(audio_raw["bytes"]))
        if orig_sr != self.target_sr:
            audio_array = _resample_audio(audio_array, orig_sr, self.target_sr)

        prompt = _PROMPT_TEMPLATE.format(
            question=row["question"],
            choice_a=row["choice_a"],
            choice_b=row["choice_b"],
            choice_c=row["choice_c"],
            choice_d=row["choice_d"],
        )
        answer_label = _CHOICE_LABELS[int(row["answer"])] if isinstance(row["answer"], int) else row["answer"]
        return {
            "audio": audio_array,
            "prompt": prompt,
            "answer": answer_label,
            "category": row.get("category", ""),
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.data[idx]
        user_content = [
            {"type": "audio", "audio": row["audio"]},
            {"type": "text", "text": row["prompt"]},
        ]
        messages = [{"role": "user", "content": user_content}]
        _datum: dict[str, Any] = dict(vlm_hf_data_processor(
            datum_dict={"messages": messages},
            task_data_spec=self.task_data_spec,
            processor=self.tokenizer,
            max_seq_length=self.max_input_seq_length,
            idx=idx,
        ))
        _datum["answer"] = row["answer"]
        _datum["category"] = row["category"]
        return _datum