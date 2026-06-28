"""
Upscalerr — Loss function definitions.

Charbonnier: Smooth L1-like loss for pixel reconstruction.
Perceptual:  VGG-16 feature-matching loss for structural fidelity.
"""

from .charbonnier import CharbonnierLoss
from .perceptual import PerceptualLoss

__all__ = ["CharbonnierLoss", "PerceptualLoss"]
