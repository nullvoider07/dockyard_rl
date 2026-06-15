# Terminal-Bench — terminal episodes

A multi-turn environment (`environments/terminal_bench_environment.py`) built on
`MultiTurnSessionEnvironment`, covering Terminal-Bench 2.1: agentic terminal tasks
graded on the **final container state**.

## The episode

The agent drives a long-lived session container, issuing one shell command per
turn (executor `/session/exec`) and observing the result. On the `TASK_COMPLETE`
signal or budget exhaustion, the episode is graded.

## Scoring

Grading is **late-injection**: the held-out `tests/` are injected only at
completion (executor `/session/finish`) — never visible to the agent during the
episode — and run against the final state. The resulting CTRF report is parsed by
`TerminalBenchReward` into the reward.

Late-injecting the tests is the integrity mechanism here, analogous to the
gold-test force-restore in the SWE path: the agent cannot read or tamper with the
grading tests because they don't exist in the container until after it has
finished acting.

Config: `examples/configs/grpo_terminal_bench.yaml`.
