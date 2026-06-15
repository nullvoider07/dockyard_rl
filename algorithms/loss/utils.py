"""Loss utility functions for Project Dockyard.

Provides masked_mean and calculate_kl, used by every loss function
and by the advantage estimator.
"""

from typing import Optional
import torch

def masked_mean(
    tensor:                     torch.Tensor,
    mask:                       torch.Tensor,
    dim:                        Optional[int] = None,
    global_normalization_factor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the mean of tensor over masked positions.

    Args:
        tensor: Values to average.
        mask:   Binary mask (1 = valid, 0 = pad).  Same shape as tensor
                or broadcastable.
        dim:    If provided, reduce along this dimension only and return a
                tensor.  If None, reduce to a scalar.
        global_normalization_factor:
                When provided, divides by this value instead of the local
                mask sum.  Used for globally-normalised losses across
                microbatches (pass the total valid token / sequence count
                across all DP ranks).

    Returns:
        Scalar or reduced tensor.
    """
    masked = tensor * mask
    if dim is not None:
        numerator   = masked.sum(dim=dim)
        denominator = mask.sum(dim=dim).clamp(min=1)
        return numerator / denominator

    if global_normalization_factor is not None:
        return masked.sum() / global_normalization_factor.clamp(min=1)

    return masked.sum() / mask.sum().clamp(min=1)

def calculate_kl(
    logprobs:           torch.Tensor,
    logprobs_reference: torch.Tensor,
    kl_type:            str = "k3",
    input_clamp_value:  Optional[float] = 20.0,
    output_clamp_value: Optional[float] = 10.0,
) -> torch.Tensor:
    """Compute a per-token KL divergence approximation.

    Supports three estimators from http://joschu.net/blog/kl-appox.html:

    k1  — log_ratio (first-order Taylor approximation)
    k2  — 0.5 * log_ratio²  (symmetric, second-order)
    k3  — exp(log_ratio) - log_ratio - 1  (Schulman approximation, always ≥ 0)

    Args:
        logprobs:           Log-probabilities from the current policy.
        logprobs_reference: Log-probabilities from the reference policy.
        kl_type:            One of "k1", "k2", "k3".
        input_clamp_value:  Clamp |log_ratio| before exponentiation to prevent
                            numerical overflow.  None to disable.
        output_clamp_value: Clamp the output KL values.  None to disable.

    Returns:
        Per-token KL tensor, same shape as logprobs.
    """
    log_ratio = logprobs - logprobs_reference

    if input_clamp_value is not None:
        log_ratio = log_ratio.clamp(-input_clamp_value, input_clamp_value)

    if kl_type == "k1":
        kl = log_ratio
    elif kl_type == "k2":
        kl = 0.5 * log_ratio ** 2
    elif kl_type == "k3":
        kl = torch.exp(log_ratio) - log_ratio - 1.0
    else:
        raise ValueError(
            f"Unknown kl_type {kl_type!r}. Valid options: 'k1', 'k2', 'k3'."
        )

    if output_clamp_value is not None:
        kl = kl.clamp(-output_clamp_value, output_clamp_value)

    return kl