"""JAX linear/hybrid attention substrate (Gated-DeltaNet) for the Qwen3-Next
family (Qwen3.5 lineage), J11.

The novel engine that the `qwen3_next` hybrid needs: a causal depthwise conv1d
short-convolution and the gated delta-rule linear-attention recurrence
(https://arxiv.org/abs/2412.06464), ported to pure JAX. The pure math lives in
`delta_rule.py` (CPU-parity-tested against the HF torch reference); the
`Qwen3NextGatedDeltaNet` NNX module wraps it with the input projections and the
gated output norm.
"""
