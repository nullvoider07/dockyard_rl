"""Sampling-param finiteness guard + multimodal input gate.

Defense-in-depth mitigations for unpatched vLLM (<= 0.23.0) advisories:

- ``assert_finite_sampling_params`` rejects NaN/Inf temperature/top_p at the
  worker boundary before any SamplingParams is built (GHSA-7h4p-rffg-7823).
- ``format_prompt_for_vllm_generation`` forwards per-sample images to the engine
  only when explicitly enabled, and EXIF/transparency-normalizes them so the
  pixels vLLM receives match the model input (GHSA-8jr5-v98p-w75m).

Imports resolve via tests/conftest.py (repo parent on sys.path + nccl stub).
"""

from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.models.generation.interfaces import assert_finite_sampling_params
from dockyard_rl.models.generation.vllm.utils import (
    _normalize_vllm_image,
    format_prompt_for_vllm_generation,
)


class TestAssertFiniteSamplingParams:
    def test_finite_passes(self):
        assert_finite_sampling_params(0.7, 0.95)
        assert_finite_sampling_params(0.0, 1.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_temperature_raises(self, bad):
        with pytest.raises(ValueError, match="temperature"):
            assert_finite_sampling_params(bad, 0.95)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_nonfinite_top_p_raises(self, bad):
        with pytest.raises(ValueError, match="top_p"):
            assert_finite_sampling_params(0.7, bad)


def _batch(token_rows, images=None):
    """Right-padded BatchedDataDict with unpadded lengths (and optional images)."""
    width = max(len(r) for r in token_rows)
    input_ids = torch.zeros((len(token_rows), width), dtype=torch.long)
    for i, row in enumerate(token_rows):
        input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    lengths = torch.tensor([len(r) for r in token_rows], dtype=torch.long)
    d = {"input_ids": input_ids, "input_lengths": lengths}
    if images is not None:
        d["vllm_images"] = images
    return BatchedDataDict(d)


class TestMultimodalGate:
    def test_text_path_strips_padding_and_needs_no_flag(self):
        out = format_prompt_for_vllm_generation(_batch([[1, 2, 3], [4, 5]]))
        assert out == [
            {"prompt_token_ids": [1, 2, 3]},
            {"prompt_token_ids": [4, 5]},
        ]
        assert all("multi_modal_data" not in p for p in out)

    def test_images_blocked_by_default(self):
        data = _batch([[1, 2]], images=[[Image.new("RGB", (2, 2))]])
        with pytest.raises(ValueError, match="allow_multimodal_inputs"):
            format_prompt_for_vllm_generation(data)

    def test_images_forwarded_when_opted_in(self):
        data = _batch([[1, 2]], images=[[Image.new("RGBA", (2, 2))]])
        out = format_prompt_for_vllm_generation(data, allow_multimodal_inputs=True)
        assert isinstance(out, list)
        images = out[0]["multi_modal_data"]["image"]
        assert len(images) == 1
        assert images[0].mode == "RGB"  # transparency flattened

    def test_no_images_present_is_unaffected_by_flag(self):
        # A sample with an empty image list takes neither branch.
        data = _batch([[7, 8]], images=[[]])
        out = format_prompt_for_vllm_generation(data, allow_multimodal_inputs=False)
        assert out == [{"prompt_token_ids": [7, 8]}]


class TestNormalizeImage:
    def test_rgba_flattened_to_rgb(self):
        assert _normalize_vllm_image(Image.new("RGBA", (3, 3))).mode == "RGB"

    def test_exif_orientation_applied(self):
        # Orientation 6 (rotate 90) swaps a non-square image's dimensions.
        base = Image.new("RGB", (4, 2))
        exif = base.getexif()
        exif[0x0112] = 6
        buf = io.BytesIO()
        base.save(buf, format="JPEG", exif=exif)
        loaded = Image.open(io.BytesIO(buf.getvalue()))
        assert loaded.size == (4, 2)
        assert _normalize_vllm_image(loaded).size == (2, 4)

    def test_non_pil_passthrough(self):
        sentinel = object()
        assert _normalize_vllm_image(sentinel) is sentinel
