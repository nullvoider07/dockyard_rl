"""Loss function interfaces for Project Dockyard.

Defines LossType, LossInputType, and the LossFunction Protocol that all
loss function implementations must satisfy.
"""

import enum
from typing import Any, Protocol
import torch
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

class LossType(enum.Enum):
    TOKEN_LEVEL    = "token_level"
    SEQUENCE_LEVEL = "sequence_level"

class LossInputType(enum.Enum):
    LOGIT       = "logit"
    LOGPROB     = "logprob"
    DISTILLATION = "distillation"
    DRAFT       = "draft"

class LossFunction(Protocol):
    """Signature for loss functions used in RL algorithms.

    Loss functions compute a scalar loss value and associated metrics from
    model logprobs and other data contained in a BatchedDataDict.
    """

    loss_type:  LossType
    input_type: LossInputType

    def __call__(
        self,
        data:               BatchedDataDict,
        global_valid_seqs:  torch.Tensor,
        global_valid_toks:  torch.Tensor,
        **kwargs:           Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute loss and metrics from logprobs and other data.

        Args:
            data:              BatchedDataDict containing rewards, values,
                               actions, advantages, masks, and other
                               algorithm-specific fields.
            global_valid_seqs: Number of valid sequences in the microbatch.
                               Used for global normalisation of sequence-level
                               losses across microbatches.
            global_valid_toks: Number of valid tokens in the microbatch.
                               Used for global normalisation of token-level
                               losses across microbatches.
            **kwargs:          Loss-function-specific inputs:
                               - LossInputType.LOGPROB:      next_token_logprobs
                               - LossInputType.LOGIT:        logits
                               - LossInputType.DISTILLATION: student_topk_logprobs,
                                                             teacher_topk_logprobs, H_all
                               - LossInputType.DRAFT:        teacher_logits,
                                                             student_logits, mask

        Returns:
            (loss, metrics) where loss is a scalar tensor to minimise and
            metrics is a dict of diagnostic values.
        """
        ...