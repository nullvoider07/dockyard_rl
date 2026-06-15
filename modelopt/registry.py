import sys

# In Dockyard all Ray actors use sys.executable directly — no uv venvs,
# no per-extra Python environments. A uv-extras invocation
# (`uv run --locked --extra modelopt --extra vllm ...`) would normally select
# the interpreter here; we collapse that to a single executable since the
# inference fleet containers already have modelopt and turboquant-plus-vllm
# installed alongside vLLM.
#
# DTensor quant workers (dtensor_quant_policy_worker*.py) are upstream stubs
# that raise NotImplementedError and are excluded.
# Quantized workers for non-DTensor backends are excluded; Dockyard is DTensor-only.

_EXECUTABLE = sys.executable

MODELOPT_ACTOR_REGISTRY: dict[str, str] = {
    "dockyard_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantGenerationWorker": _EXECUTABLE,
    "dockyard_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantAsyncGenerationWorker": _EXECUTABLE,
}