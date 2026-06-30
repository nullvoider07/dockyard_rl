# Quantized generation

Rollout generation can serve a **quantized** model while the trainer stays in
bf16, so the rollout fleet runs faster and lighter without quantizing the
training math. The wrinkle is RL refit: fresh weights stream from the trainer to
the inference engine every step, so the quantization must be **re-applied each
refit**, not once at load. Two paths exist — block-wise FP8 and ModelOpt NVFP4.

## Block-wise FP8

vLLM serves a block-wise-FP8 model (`models/generation/vllm/quantization/fp8.py`);
the bf16 weights from the trainer are quantized to FP8 on each refit, including
the per-expert MoE path. Training numerics are untouched — only the served copy
is FP8.

## ModelOpt NVFP4 real-quant

dockyard's ModelOpt path defaults to **fakequant**: a hook that simulates NVFP4's
numerics on a full-precision kernel, useful for studying the accuracy impact
without the real kernel. **Real-quant** (`cfg.real_quant=true`) instead serves a
true NVFP4 W4A16 model through vLLM's native FP4 (Marlin) kernel, setting
`quantization="modelopt"` and the `VLLM_MODELOPT_REAL_QUANT` env so vLLM loads the
FP4 method.

The challenge real-quant solves is refit-repeatability. vLLM's
`process_weights_after_loading` converts the loaded HF-named params (`weight` /
`weight_scale` / `weight_scale_2`) into the kernel layout **once** — deleting and
renaming params and dropping `weight_loader` references along the way. That is
fine for a static deployment but breaks RL, where new NVFP4 weights must be
re-loadable every step. The patches in
`modelopt/models/generation/vllm_modelopt_patch.py` make the conversion
repeatable:

- before converting, the loaded param shape/dtype/loader are captured so a
  subsequent refit can restore loadable parameters
  (`prepare_modelopt_for_weight_reload`),
- a weight-only (W4A16) checkpoint is tagged and routed to the Marlin FP4 GEMM,
- after a refit streams new weights, the conversion is re-run
  (`modelopt_process_weights_after_loading`).

**Scope.** This is the *inference-side* real-quant path: it serves an NVFP4
checkpoint and keeps it loadable across refits. A trainer-side DTensor NVFP4
**producer** (quantizing the policy's weights to NVFP4 before sync) is a
documented integration seam rather than a built component — the only existing
upstream producer is tied to a model-parallel training stack outside dockyard's
DTensor/FSDP2 design. Until that seam is wired, real-quant serves an
externally-produced NVFP4 model.

## Validation

The config switch, the patch wiring, and the W4A16 detection are CPU unit-tested
with the engine mocked. Live NVFP4 serving and the refit re-quantization round
trip are GPU-only and tracked in the hardware-deferred-validation ledger.
