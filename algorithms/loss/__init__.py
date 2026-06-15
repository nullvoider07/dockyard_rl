"""dockyard_rl.algorithms.loss — loss functions and interfaces."""

from dockyard_rl.algorithms.loss.interfaces import LossFunction, LossInputType, LossType
from dockyard_rl.algorithms.loss.loss_functions import (
    ClippedPGLossConfig,
    ClippedPGLossDataDict,
    ClippedPGLossFn,
    DistillationLossConfig,
    DistillationLossDataDict,
    DistillationLossFn,
    CDPOLossFn,
    DPOLossConfig,
    DPOLossDataDict,
    DPOLossFn,
    DPOPLossFn,
    IPOLossFn,
    KTOLossFn,
    ORPOLossFn,
    RDPOLossFn,
    SimPOLossFn,
    DraftCrossEntropyLossConfig,
    DraftCrossEntropyLossDataDict,
    DraftCrossEntropyLossFn,
    NLLLossFn,
    PREFERENCE_LOSS_REGISTRY,
    PreferenceLossDataDict,
    PreferenceLossFn,
    build_preference_loss,
    is_reference_free,
)
from dockyard_rl.algorithms.loss.utils import calculate_kl, masked_mean

__all__ = [
    # Interfaces
    "LossFunction", "LossType", "LossInputType",
    # Loss functions
    "ClippedPGLossConfig", "ClippedPGLossDataDict", "ClippedPGLossFn",
    "NLLLossFn",
    "PreferenceLossDataDict", "PreferenceLossFn",
    "DPOLossConfig", "DPOLossDataDict", "DPOLossFn",
    "CDPOLossFn", "DPOPLossFn", "RDPOLossFn", "IPOLossFn", "KTOLossFn",
    "SimPOLossFn", "ORPOLossFn",
    "PREFERENCE_LOSS_REGISTRY", "build_preference_loss", "is_reference_free",
    "DistillationLossConfig", "DistillationLossDataDict", "DistillationLossFn",
    "DraftCrossEntropyLossConfig", "DraftCrossEntropyLossDataDict", "DraftCrossEntropyLossFn",
    # Utilities
    "masked_mean", "calculate_kl",
]