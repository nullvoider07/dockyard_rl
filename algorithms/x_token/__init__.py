"""Cross-tokenizer (xtoken) off-policy distillation.

Distill from a teacher whose tokenizer differs from the student's: a TokenAligner
maps teacher tokens to student tokens, the teacher's full-vocab logits are
projected onto the student vocab via a sparse projection, and a chunk-averaged
cross-tokenizer KL forms the loss. See ``token_aligner`` (alignment), ``utils``
(batch-grid / transport guards), and ``loss_utils`` (projection + chunk math +
teacher-logit transport).
"""
