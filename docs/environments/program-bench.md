# ProgramBench — cleanroom reconstruction

A multi-turn environment (`environments/program_bench_environment.py`) built on
`MultiTurnSessionEnvironment`. The task is **cleanroom reconstruction**: the agent
is given an execute-only reference binary (`./executable`) and must reproduce its
behaviour from scratch.

## The episode

The agent drives a long-lived session container in a **network-isolated** session
(with `SYS_PTRACE` dropped, so it cannot inspect the binary's internals — it can
only probe behaviour). Each turn it issues one shell command — running the
reference binary on inputs, writing source files, writing a `compile.sh` — and
observes the output. On the `TASK_COMPLETE` signal or the turn budget, the
agent's `/workspace` is exported as the submission.

## Scoring

`ProgramBenchReward` grades the submission by **rebuilding it from scratch in a
clean image** (via `compile.sh`) and running the held-out behavioural pytest
branches against the rebuilt program. Because grading rebuilds in a fresh
container, the agent cannot smuggle the reference binary or build artifacts into
the submission — only reconstructed source that compiles and behaves correctly
scores.

Config: `examples/configs/grpo_program_bench.yaml`. The cleanroom probe-only
constraint (network isolation, no ptrace) is what makes the reconstruction
genuine rather than a copy.
