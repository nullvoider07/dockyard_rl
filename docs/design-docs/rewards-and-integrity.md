# Rewards and integrity

The reward is the load-bearing part of an RL system — if it can be gamed, the
policy will game it. dockyard_rl grounds the reward in **real test execution**
and hardens the path against the specific ways an agent can cheat a
patch-and-test loop.

## The reward contract

Every reward implements `RewardFunction` (`rewards/interfaces.py`): it takes a
trajectory dict and returns a `RewardVerificationResult` — a named tuple of
`(reward, status, evidence_hash, failure_reason)`. The status distinguishes a
clean score (`ok`) from an `integrity_failure` or an `execution_error`, and the
evidence hash pins the test output that produced the score.

For SWE scoring the trajectory dict carries the scoring inputs: `sandbox_url`,
`task_id`, `repo_url`, `base_commit`, the agent's `patch`, the gold `test_patch`,
the `fail_to_pass` / `pass_to_pass` node IDs, and the reference `gold_patch`
(used only for patch-similarity diagnostics, never for scoring).

## Test-runner scoring

`TestRunnerReward` (`rewards/test_runner.py`) scores a solution through the
sandbox executor's `/task/submit` API:

1. The agent `patch` is applied to a fresh clone at `base_commit`.
2. The executor applies the gold `test_patch`.
3. The `FAIL_TO_PASS ∪ PASS_TO_PASS` node IDs are run under pytest
   (`python3 -m pytest -p no:cacheprovider -q …`).
4. The task is **resolved** iff the command exits 0 with no failures and at least
   the expected number of tests pass.

`reward_mode` selects the shape: `binary` (1.0 resolved / 0.0 otherwise) or
`test_pass_rate` (the pass fraction). Sibling rewards cover the other
environments — `SWEBenchProReward`, `program_bench`, `terminal_bench`,
`hle_grader`, `gdpval_rubric` — and `compiler` / `diff_utils` provide the shared
patch-parsing and build primitives.

## The integrity path

The obvious exploit in a patch-and-test loop is to edit the tests. Two
independent mechanisms make that structurally inert:

1. **Executor force-restore.** Before running, the executor force-applies the
   gold `test_patch`, so the held-out test files are canonical regardless of
   what the agent's patch did to them. A patch that rewrites a test cannot change
   what actually runs.
2. **`IntegrityReward`** (`rewards/integrity.py`). This decorator wraps an inner
   reward and zeroes it whenever the agent patch touches a held-out test path.
   The held-out paths are exactly the files the gold `test_patch` modifies;
   detection is a **static comparison of the two diffs' touched paths**
   (`parse_diff_paths`) with no executor call. It catches editing an existing
   test file, replacing its contents, or adding a file at a held-out test path,
   and reports `integrity_failure`.

Force-restore makes tampering ineffective; `IntegrityReward` makes it
*penalized* — the agent both fails to influence the verdict and scores zero, so
there is no gradient signal rewarding the attempt. Enable it per environment with
`env.code.integrity_check`.

## The invalid-action penalty

Beyond test tampering, `rewards/invalid_action.py` applies a configurable reward
penalty for malformed agent behaviour — invalid tool calls and malformed
thinking blocks — sourced either from env-authoritative per-turn verdicts or a
generic pattern fallback. Disabled (the default) is a strict no-op: no detection
runs and rewards are untouched. It shares its signal with the
[tool-use protocol](tool-use-protocol.md) validator.
