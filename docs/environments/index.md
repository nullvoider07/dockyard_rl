# Environments

An environment turns a model's output into a reward. dockyard_rl ships a family
of them — from single-shot patch scoring to multi-turn agentic sessions and full
computer-use — all behind one interface, so the GRPO loop drives any of them
without special-casing.

## The interface

Every environment implements `EnvironmentInterface` (`environments/interfaces.py`):

- `step(...)` — given the agent's output (and per-episode metadata), produce an
  `EnvironmentReturn`: the next observation, the reward, a termination flag, and
  forward-threaded metadata. Single-shot environments terminate after one step;
  multi-turn ones return the next observation and carry session state forward in
  the metadata channel.
- `global_post_process_and_metrics(...)` — batch-level post-processing and metric
  aggregation after a generation pass.

An environment is selected per dataset (`data.default.env_name`) and configured
under the `env` config section.

## Two shapes

| Shape | Pattern |
| --- | --- |
| **Single-shot** | The agent emits one output, scored once, episode ends. Used for SWE patch scoring and the verifier-style evals (HLE, math). |
| **Multi-turn session** | `MultiTurnSessionEnvironment` is the shared base: the agent drives a long-lived per-episode `ubuntu-swe` container, issuing one command per turn (executor `/session/exec`), until a `TASK_COMPLETE` signal or the turn budget, then the result is graded (`/session/finish`). Used by program-bench, terminal-bench, and the GDPval agentic path. |

The generic rollout loop (`experience/rollouts.py`) drives both: it calls
`step()` each turn, tokenizes the returned observation as the next user message,
and threads the returned metadata forward.

## The environments

| Environment | Page | Shape |
| --- | --- | --- |
| SWE-bench / SWE-bench Pro | [SWE](swe.md) | single-shot |
| ProgramBench | [ProgramBench](program-bench.md) | multi-turn session |
| Terminal-Bench | [Terminal-Bench](terminal-bench.md) | multi-turn session |
| OSWorld (computer-use) | [OSWorld / CUA](osworld-cua.md) | multi-turn, GUI |
| GDPval | [GDPval](gdpval.md) | text + agentic |
| HLE, math | [Verifier evals](verifier-evals.md) | single-shot |

```{toctree}
:hidden:
:maxdepth: 1

swe
program-bench
terminal-bench
osworld-cua
gdpval
verifier-evals
```
