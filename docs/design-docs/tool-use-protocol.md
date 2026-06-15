# Structured tool-use protocol

By default, agents emit actions as fenced text (a ` ```diff ` block, a `<patch>`
tag). The structured tool-use protocol (`tool_protocol/`) is an opt-in upgrade
that lets the model emit **native Hermes `<tool_call>` JSON** instead — the same
format Qwen produces under vLLM serving — so a model trained here and a model
served by vLLM see identical structure.

The whole protocol is **disabled by default and a strict no-op when off**: the
fenced-text rollout, prompt, and parse paths run byte-identically. It activates
only when both `grpo.structured_tool_use.enabled` and the environment's
`structured_tools` flag are set.

## Pieces

| Module | Role |
| --- | --- |
| `hermes.py` | Extracts `<tool_call>{json}</tool_call>` blocks, mirroring vLLM's `Hermes2ProToolParser` but with no OpenAI-protocol dependency and richer per-call parse detail. |
| `schema.py` | A dependency-free JSON-schema validator for tool-call arguments. |
| `registry.py` | The per-environment tool registry — the typed tools an environment advertises. |
| `protocol.py` | Config (`StructuredToolUseConfig`) and the resolution that wires the above into the rollout and data layers. |

These are pure, CPU-only building blocks; the rollout/penalty wiring layers on
top of them.

## How it threads through a run

1. The environment's tool registry is advertised in the prompt, so the model
   knows the typed tools available (for SWE, `submit_patch`).
2. The agent emits typed `<tool_call>` JSON. The rollout wraps tool results in
   the Qwen turn envelope.
3. The extractor parses each call; the schema validator checks arguments against
   the tool's schema.
4. Validation feeds the [invalid-action penalty](rewards-and-integrity.md): a
   malformed or schema-invalid call becomes a reward signal rather than a silent
   failure.

`thinking_style` (`paired` for Qwen2.5/older Qwen3, `closing_only` for Qwen3.5)
and `decode_skip_special_tokens` adapt the envelope to the model generation.

## RL-safe constraining

vLLM's `structural_tag` can *constrain* generation to the tool grammar, but
constraining is dangerous in RL: forced tokens carry no policy signal, and
training on them would teach the model nothing while corrupting the loss. The
protocol keeps it RL-safe — only the **scaffolding** is grammar-forced
(the `<tool_call>` envelope), those forced tokens are **loss-masked**, and the
argument values and thinking stay free generation. `constrained_tools` selects
which tools are constrained; the recommended default constrains only the
single-shot patch submission, leaving everything else free-gen with post-hoc
validation.
