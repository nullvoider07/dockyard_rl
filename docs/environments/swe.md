# SWE — coding-agent patch scoring

The flagship environment (`environments/code_environment.py`). The episode is
single-shot: the agent emits a unified-diff solution, which the environment
submits to a sandbox task executor (`POST /task/submit`). The executor clones the
repo fresh, applies the agent patch, applies the gold `test_patch`, runs the
`FAIL_TO_PASS` / `PASS_TO_PASS` tests, and returns the verdict.

## Scoring

The reward comes from `TestRunnerReward`, wrapped by `IntegrityReward` (when
`env.code.integrity_check` is on) so a patch that edits a held-out test file
scores zero. See [rewards and integrity](../design-docs/rewards-and-integrity.md)
for the scoring and anti-tampering details, and [the sandbox executor](../design-docs/sandbox-executor.md)
for the execution side.

`env.code.reward_mode` selects `binary` (resolved / not) or `test_pass_rate` (the
pass fraction).

## Data

Each sample's `extra_env_info` (produced by `swe_bench_data_processor`) carries
`repo`, `base_commit`, `fail_to_pass`, `pass_to_pass`, `test_patch`, and
`gold_patch`. The executor is stateless per task, so no per-episode provisioning
is needed; endpoints come from `env.code.sandbox_urls` (or `DOCKYARD_SANDBOX_URLS`)
and are round-robined across the batch.

Datasets: `swe_bench` (SWE-bench / SWE-bench Lite) and `swe_bench_pro`
(`SWEBenchProReward`). The configs are `examples/configs/grpo_swe.yaml`,
`grpo_swe_pro.yaml`, and `grpo_swe_sglang.yaml` (SGLang backend).

## Structured tools

With `env.code.structured_tools: true` (paired with
`grpo.structured_tool_use.enabled`), the solution is parsed as a structured
`submit_patch` tool call (Hermes `<tool_call>`) instead of a ` ```diff ` fence or
`<patch>` tag. Off is byte-identical to the fenced-text path. See the
[tool-use protocol](../design-docs/tool-use-protocol.md).
