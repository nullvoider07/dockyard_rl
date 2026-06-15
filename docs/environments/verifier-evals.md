# Verifier evals — HLE and math

Two single-turn verifier environments. The agent answers once, the answer is
extracted and graded, and the episode ends. Both are primarily **validation**
signals rather than training targets.

## HLE (Humanity's Last Exam)

`environments/hle_environment.py` — single-turn, no sandbox. The agent answers in
the fixed HLE format (Explanation / Answer / Confidence); the extracted answer is
graded against the gold answer by an LLM judge when one is configured, falling
back to a normalized exact-match otherwise (`rewards/hle_grader.py`).

Intended as a validation benchmark — judge cost is bounded to validation passes.
Training on the 2,500-question exam is not recommended. Config:
`examples/configs/grpo_hle.yaml`.

## Math

`environments/math_environment.py` — single-turn. The model's answer is verified
with `math_verify` (symbolic/numeric equivalence grading rather than string
match), so algebraically-equivalent answers in a different form still score. A
verification timeout is handled gracefully.

## Related

`environments/code_jaccard_environment.py` scores a diff by token-Jaccard
similarity to a reference (a cheap, judge-free proxy), and
`environments/reward_model_environment.py` scores with a learned reward model for
the RLAIF / reward-model paths. Both follow the same single-shot
`EnvironmentInterface` contract.
