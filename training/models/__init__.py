"""
Upscalerr — Neural network model definitions.

ESPCN: Efficient Sub-Pixel Convolutional Network for spatial upscaling.
FlowNet: Lightweight optical-flow U-Net for frame interpolation.
"""

from .espcn import ESPCN
from .flownet import FlowNetLite

__all__ = ["ESPCN", "FlowNetLite"]
