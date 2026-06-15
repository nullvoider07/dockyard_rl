"""GDPval dataset loader for Project Dockyard (text-deliverable path).

Trainer-side only. Loads ``openai/gdpval`` (220 gold tasks) and presents each as a
``messages`` + metadata row, graded by GDPvalRubricReward (an LLM judge scores the
model's TEXT deliverable against the task's ``rubric_json``).

GDPval is fundamentally a file-producing benchmark (deliverables are PDFs/
spreadsheets/etc.; ~68% of tasks ship binary reference files). This loader is the
reduced text-only path: by default it keeps only tasks with NO reference files
(``require_no_reference_files=True``), since a text policy cannot consume binary
inputs. File/format rubric criteria are unsatisfiable by text output — the
faithful path is the agentic file-producing environment (deferred; see handoff).

Schema (openai/gdpval, ``train`` split): task_id, sector, occupation, prompt,
reference_files[/_urls/_hf_uris], deliverable_files[/_urls/_hf_uris],
rubric_pretty, rubric_json.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Optional

from datasets import load_dataset
from dockyard_rl.data.datasets.raw_dataset import RawDataset

_SYSTEM_PROMPT = """\
You are an experienced professional completing a real work task. Produce the
complete deliverable the task asks for.

Your output is text, so present the full content directly: write documents in
Markdown, and represent any spreadsheet/table as a Markdown table with every row,
each formula written out as text, and the computed values shown explicitly. State
all assumptions, methods, and numbers — your deliverable is graded against a
detailed professional rubric, so be specific and complete rather than summarising."""

_AGENTIC_SYSTEM_PROMPT = """\
You are an experienced professional completing a real work task inside a Linux
sandbox. Produce the complete deliverable as actual FILES.

Each turn, reply with exactly one shell command in a single ```bash code block;
its output is returned to you. You have python3 with openpyxl, python-docx,
python-pptx, reportlab, pandas, matplotlib and pypdf, plus pandoc. Build the
deliverable as real files (e.g. .xlsx/.docx/.pptx/.pdf/.md) under
{deliverable_dir}/ — create that directory if needed. Any provided reference
files are under {reference_dir}/.

Work step by step. When the deliverable is complete and saved under
{deliverable_dir}/, reply with TASK_COMPLETE on its own line. Your produced files
are graded against a detailed professional rubric, so satisfy every criterion
(file formats, named sheets/sections, computed values) explicitly."""


def _encode_reference_files(row: dict) -> dict[str, str]:
    """Best-effort base64 encoding of a row's inline binary reference files.

    GDPval rows carry ``reference_files`` as binary inputs; HF representations
    vary (list of dicts with ``bytes``/``filename``/``path``, or {name: bytes}).
    Inline bytes are encoded; URI-only entries (no bytes available offline) are
    skipped. Returns {filename: base64}.
    """
    refs = row.get("reference_files")
    out: dict[str, str] = {}
    if not refs:
        return out
    items: list = refs if isinstance(refs, list) else [refs]
    for i, entry in enumerate(items):
        name: Optional[str] = None
        data: Optional[bytes] = None
        if isinstance(entry, dict):
            name = entry.get("filename") or entry.get("name") or entry.get("path")
            raw = entry.get("bytes")
            if isinstance(raw, (bytes, bytearray)):
                data = bytes(raw)
        elif isinstance(entry, (bytes, bytearray)):
            data = bytes(entry)
        if data is None:
            continue
        fname = str(name) if name else f"reference_{i}"
        out[fname] = base64.b64encode(data).decode("ascii")
    return out


def _has_reference_files(row: dict) -> bool:
    refs = row.get("reference_files") or []
    return len(refs) > 0


class GDPvalDataset(RawDataset):
    """GDPval tasks exposed as ``messages`` + metadata rows (text-deliverable path).

    Args:
        hf_dataset_name: HuggingFace dataset name or local path.
        split:           Dataset split (default: "train"; the gold set is 220 tasks).
        require_no_reference_files: Keep only tasks with no reference files
                         (default True — the text path cannot consume binary inputs).
        occupations:     Optional subset of occupation names.
        task_ids:        Optional subset of task ids.
        shuffle_seed:    If set, shuffle with this seed.
    """

    def __init__(
        self,
        hf_dataset_name: str = "openai/gdpval",
        split: str = "train",
        require_no_reference_files: bool = True,
        occupations: Optional[list[str]] = None,
        task_ids: Optional[list[str]] = None,
        shuffle_seed: Optional[int] = None,
        agentic: bool = False,
        image: str = "",
        deliverable_dir: str = "/workspace/deliverable",
        reference_dir: str = "/workspace/reference",
        max_turns: int = 30,
        exec_timeout_sec: int = 120,
        provision_reference_files: bool = False,
        **kwargs: Any,
    ) -> None:
        self.task_name = "gdpval"
        self.agentic = bool(agentic)
        self.image = image
        self.deliverable_dir = deliverable_dir
        self.reference_dir = reference_dir
        self.max_turns = int(max_turns)
        self.exec_timeout_sec = int(exec_timeout_sec)
        self.provision_reference_files = bool(provision_reference_files)

        print(f"Loading GDPval dataset: {hf_dataset_name} (split={split})", flush=True)
        raw = load_dataset(hf_dataset_name, split=split)

        if task_ids:
            id_set = set(task_ids)
            raw = raw.filter(lambda x: x.get("task_id") in id_set)
        if occupations:
            occ_set = set(occupations)
            raw = raw.filter(lambda x: x.get("occupation") in occ_set)

        if require_no_reference_files:
            n_before = len(raw)
            raw = raw.filter(lambda x: not _has_reference_files(x))
            dropped = n_before - len(raw)
            if dropped:
                print(
                    f"  ⚠ Dropped {dropped} task(s) requiring binary reference files "
                    "(text path); set require_no_reference_files=false to keep them",
                    flush=True,
                )

        if shuffle_seed is not None:
            raw = raw.shuffle(seed=shuffle_seed)

        self.dataset = raw.map(self.format_data, remove_columns=raw.column_names)
        self.val_dataset = None
        if len(self.dataset) == 0:
            raise ValueError(
                "No GDPval tasks after filtering. With require_no_reference_files=True "
                "only the no-reference-file subset is kept; relax the filter or the "
                "occupation/task_id selection."
            )
        print(f"  ✓ Loaded {len(self.dataset)} GDPval tasks", flush=True)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        prompt = (data.get("prompt") or "").strip()
        if not self.agentic:
            return {
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "task_name":   self.task_name,
                "task_id":     data.get("task_id", ""),
                "occupation":  data.get("occupation", ""),
                "sector":      data.get("sector", ""),
                # Held-out scoring inputs routed into extra_env_info by the processor.
                "prompt":      prompt,
                "rubric_json": data.get("rubric_json") or "",
            }

        # Agentic (file-producing) path: a tool-use system prompt + the session
        # metadata the multi-turn environment needs. Variable-key structs
        # (reference files) are carried as a JSON string (Arrow cannot hold them).
        reference_files = (
            _encode_reference_files(data) if self.provision_reference_files else {}
        )
        system_prompt = _AGENTIC_SYSTEM_PROMPT.format(
            deliverable_dir=self.deliverable_dir,
            reference_dir=self.reference_dir,
        )
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            "task_name":           self.task_name,
            "task_id":             data.get("task_id", ""),
            "occupation":          data.get("occupation", ""),
            "sector":              data.get("sector", ""),
            "prompt":              prompt,
            "rubric_json":         data.get("rubric_json") or "",
            "image":               self.image,
            "deliverable_dir":     self.deliverable_dir,
            "reference_dir":       self.reference_dir,
            "max_turns":           self.max_turns,
            "exec_timeout_sec":    self.exec_timeout_sec,
            "reference_files_json": json.dumps(reference_files),
        }


class GDPvalAgenticDataset(GDPvalDataset):
    """GDPval tasks for the agentic file-producing path (multi-turn sandbox).

    Thin specialisation of :class:`GDPvalDataset` with ``agentic=True``: the model
    is prompted to produce real deliverable FILES in a container, graded by the
    same rubric judge over their extracted text. ``image`` is the file-tooling
    sandbox image (``ubuntu-swe-gdpval``). ``provision_reference_files`` (default
    False) controls whether inline binary reference files are staged into the
    container; with it False the loader keeps the no-reference-file subset.
    """

    def __init__(
        self,
        hf_dataset_name: str = "openai/gdpval",
        split: str = "train",
        image: str = "",
        require_no_reference_files: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            hf_dataset_name=hf_dataset_name,
            split=split,
            image=image,
            require_no_reference_files=require_no_reference_files,
            agentic=True,
            **kwargs,
        )
