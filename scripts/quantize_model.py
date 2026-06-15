"""Offline post-training quantization script for dockyard_rl.

Quantizes a HuggingFace AutoModelForCausalLM checkpoint with ModelOpt
(nvidia-modelopt) and saves the result as HuggingFace safetensors so
that vLLM can load it directly with --load-format auto.

Typical usage
-------------
# FP8 weight quantization (recommended for H100, Ada Lovelace):
python scripts/quantize_model.py \\
    --model  /checkpoints/qwen3-72b \\
    --output /checkpoints/qwen3-72b-fp8 \\
    --quant  FP8_DEFAULT_CFG

# NVFP4 (highest compression, H100 SXM5 / Blackwell):
python scripts/quantize_model.py \\
    --model  /checkpoints/llama4-scout \\
    --output /checkpoints/llama4-scout-nvfp4 \\
    --quant  NVFP4_DEFAULT_CFG \\
    --calib-size 256

# Custom recipe from a YAML file:
python scripts/quantize_model.py \\
    --model  /checkpoints/deepseek-v3 \\
    --output /checkpoints/deepseek-v3-fp8-kv \\
    --quant  general/ptq/nvfp4_default-fp8_kv

The output directory contains:
  config.json          (updated with quantization_config block)
  model*.safetensors   (quantized weights with folded scales)
  tokenizer*           (copied from source)
  quantization.json    (ModelOpt quant config snapshot for reproducibility)

This checkpoint is then referenced as VllmConfig.model_name in your
training YAML together with VllmConfig.quant_cfg set to the same
quant string so vLLM initialises the FakeQuantWorker correctly.
"""

import argparse
import json
import os
import sys

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline ModelOpt PTQ for dockyard_rl inference fleet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local path to load from.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Directory to write the quantized checkpoint into.",
    )
    p.add_argument(
        "--quant",
        default="FP8_DEFAULT_CFG",
        help=(
            "Quantization config string passed to resolve_quant_cfg(). "
            "Examples: FP8_DEFAULT_CFG, NVFP4_DEFAULT_CFG, "
            "general/ptq/nvfp4_default-fp8_kv, or a path to a YAML recipe."
        ),
    )
    p.add_argument(
        "--calib-size",
        type=int,
        default=128,
        help="Number of calibration samples.",
    )
    p.add_argument(
        "--calib-dataset",
        default="cnn_dailymail",
        choices=["cnn_dailymail", "random"],
        help=(
            "Calibration dataset. 'cnn_dailymail' uses real text from the "
            "CNN/DailyMail summarization dataset (requires internet access). "
            "'random' uses random token ids (offline, less accurate)."
        ),
    )
    p.add_argument(
        "--calib-max-length",
        type=int,
        default=512,
        help="Token length of each calibration sample.",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Master weight dtype before quantization.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Device to load the model on.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True to from_pretrained.",
    )
    return p.parse_args()

def _build_calib_dataloader(
    dataset_name: str,
    tokenizer,
    calib_size: int,
    max_length: int,
    device: str,
):
    """Return a list of token-id batches for calibration forward passes."""
    import torch

    if dataset_name == "random":
        vocab_size = tokenizer.vocab_size or 32000
        return [
            {
                "input_ids": torch.randint(
                    0, vocab_size, (1, max_length), device=device
                )
            }
            for _ in range(calib_size)
        ]

    # cnn_dailymail — load a small slice for calibration
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for --calib-dataset cnn_dailymail. "
            "Install with: pip install datasets"
        )

    ds = load_dataset("cnn_dailymail", "3.0.0", split="train")
    texts = [ds[i]["article"] for i in range(min(calib_size * 4, len(ds)))]

    batches = []
    for text in texts:
        ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ).input_ids.to(device)
        batches.append({"input_ids": ids})
        if len(batches) >= calib_size:
            break

    return batches

def main() -> None:
    args = _parse_args()

    try:
        import torch
        import modelopt.torch.quantization as mtq
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        sys.exit(
            f"Missing dependency: {exc}\n"
            "Install with: pip install nvidia-modelopt[torch] transformers"
        )

    # Resolve quantization config
    from dockyard_rl.modelopt.utils import resolve_quant_cfg
    quant_cfg = resolve_quant_cfg(args.quant)
    print(f"Quantization config resolved: {args.quant}")
    print(f"  quant_cfg keys: {list(quant_cfg.keys())}")

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    # Load model in master dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    print(f"Loading model from {args.model} in {args.dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    # Build calibration data
    print(
        f"\nBuilding calibration set: dataset={args.calib_dataset}, "
        f"size={args.calib_size}, max_length={args.calib_max_length} ..."
    )
    calib_batches = _build_calib_dataloader(
        args.calib_dataset,
        tokenizer,
        args.calib_size,
        args.calib_max_length,
        args.device,
    )
    print(f"  Calibration batches ready: {len(calib_batches)}")

    # Calibration forward loop
    def forward_loop(model):
        for batch in calib_batches:
            with torch.no_grad():
                model(**batch)

    # Quantize
    print(f"\nRunning mtq.quantize() with config: {args.quant} ...")
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    mtq.print_quant_summary(model)

    # Fold quantizer scales into weights for vLLM loading
    print("\nFolding weight scales ...")
    mtq.fold_weight(model)

    # Save output
    os.makedirs(args.output, exist_ok=True)
    print(f"\nSaving quantized checkpoint to {args.output} ...")
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)

    # Save a quantization metadata snapshot alongside the checkpoint
    quant_meta = {
        "quant_cfg_name": args.quant,
        "quant_cfg":      quant_cfg,
        "calib_dataset":  args.calib_dataset,
        "calib_size":     args.calib_size,
        "calib_max_length": args.calib_max_length,
        "source_model":   args.model,
        "dtype":          args.dtype,
    }
    with open(os.path.join(args.output, "quantization.json"), "w") as f:
        json.dump(quant_meta, f, indent=2, default=str)

    print(f"\nDone. Quantized checkpoint saved to: {args.output}")
    print(
        "\nTo use this checkpoint in your training YAML:\n"
        f"  vllm_cfg:\n"
        f"    model_name: {args.output}\n"
        f"  quant_cfg: {args.quant}\n"
    )

if __name__ == "__main__":
    main()