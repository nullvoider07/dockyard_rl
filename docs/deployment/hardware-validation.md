# Hardware-deferred validation

dockyard_rl is validated without a GPU cluster wherever possible: `py_compile`, a clean
`pyright` pass, and pure-Python / numeric unit tests. Some behavior is
**correct-by-design and statically validated but cannot be confirmed without a GPU or a
multi-node cluster** — collective weight transfers, expert-parallel MoE sharding and
grouped GEMMs, FP8 and GGUF kernels, the data plane, and the live generation-engine
paths. Those slices are tracked and must be exercised at bring-up.

The dropdown below summarizes, by area, **what is not yet confirmed on hardware**, so
you know which paths to validate first when bringing up a real cluster. Each item is
disabled-by-default or byte-identical when its feature is off; the table marks only the
slice that needs a GPU. The full per-item ledger — with the exact assertion to run for
each — lives in `handoff/hardware-deferred-validation.md` (ledger IDs in parentheses).

<details>
<summary><strong>GPU validations not yet confirmed</strong> (click to expand)</summary>

| Area | Not yet confirmed on hardware |
| --- | --- |
| MoE expert parallelism | Device-mesh build and rank→expert mapping, `distribute_tensor` placements, all-to-all token dispatch, EP + FSDP composition, the CUDA-only grouped-GEMM expert forward, EP refit gather, load-balance reduction, and the model surgery on a real checkpoint (HV-1–3, 8–15). |
| Weight sync / refit | The NCCL-collective, CUDA-IPC, and SGLang-HTTP weight transports plus the GPU offload / prepare-for-generation transitions (HV-7). |
| FP8 serving | Block-FP8 grouped-GEMM forward, per-expert `_scale_inv` loading into the FusedMoE, and per-refit re-quantization on Hopper/Blackwell (HV-16). |
| Data plane (TransferQueue) | The two-phase NCCL broadcast on a real replica group, leader-only exactly-once write-back, and async dispatch (HV-17, 18). |
| Generation-engine internals | The vLLM private-attribute send-lock path, generation-logprob presence on every assistant message, and master-port-range propagation to workers (HV-4–6). |
| Structured tool-use | Turn-envelope bytes vs the real chat template, constrained-decoding compile + the forced-skeleton submask, and live env tool dispatch (HV-21–23). |
| Invalid-action penalty | The live per-turn verdict wiring through a real `env.step` + reward round-trip (HV-19). |
| GDPval agentic env | File production in a live `ubuntu-swe-gdpval` container and the binary document extractors (HV-20). |
| Preference optimization (KTO) | The worker-side KL reference `z` from mismatched completions and the unpaired-batch driver path (HV-24). |
| Image multimodal (#50) | A live vLLM VLM run forwarding the normalized images; SGLang multimodal is **gate-and-surface only** — forwarding is not implemented and needs a live SGLang engine to build and validate (HV-36). |
| GGUF export + serve (#51) | Live llama.cpp conversion of a real checkpoint; the #51 dequant guard is built but **not yet wired** — the static GGUF-in-vLLM serve mode is future work. Export + validation never invoke vLLM, so #51 is unexercised there (HV-37). |

</details>

When a feature above is **off** (the default for the opt-in ones), its code path is
byte-identical to the validated baseline. Turning one on for the first time on real
hardware should be paired with the corresponding bring-up check from the ledger.
