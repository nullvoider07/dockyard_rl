# GDPval — file-producing deliverables

GDPval is a set of 220 long-horizon, file-oriented professional tasks. dockyard_rl
has two paths, sharing one judge.

## Text path

`environments/gdpval_environment.py` — single-turn, no sandbox. The model produces
a textual deliverable; it is scored against the task's `rubric_json` by an LLM
judge (`GDPvalRubricReward`). The episode ends after this one scoring step.

Rubric grading is **judgmental with no deterministic fallback** — a judge
endpoint must be configured, or every sample is an `execution_error`. This is the
reduced path, intended primarily as an eval/validation signal.

## Agentic path

`environments/gdpval_agentic_environment.py` — multi-turn, sandbox-backed, the
faithful path. Built on `MultiTurnSessionEnvironment`, the agent drives a
long-lived container and uses shell/python tools to **produce real deliverable
files** (xlsx, docx, pptx, pdf, …) under `/workspace/deliverable`, emitting one
` ```bash ` block per turn (or `TASK_COMPLETE` to finish).

On completion or budget exhaustion, the produced files are extracted to text
in-container (a manifest plus per-file content) and graded with the **same**
`GDPvalRubricReward` as the text path — so file- and format-oriented rubric
criteria that the text path structurally cannot satisfy become demonstrable to
the judge.

Configs: `examples/configs/grpo_gdpval.yaml` (text) and `grpo_gdpval_agentic.yaml`
(agentic).
