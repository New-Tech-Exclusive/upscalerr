"""
Upscalerr — Dataset definitions.

UpscaleDataset: DIV2K HR/LR pair dataset for ESPCN training.
FlowDataset:   Synthetic motion triplet dataset for FlowNet training.
"""

from .upscale_dataset import UpscaleDataset
from .flow_dataset import FlowDataset

__all__ = ["UpscaleDataset", "FlowDataset"]
