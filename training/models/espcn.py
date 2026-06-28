"""
Efficient Sub-Pixel Convolutional Network (ESPCN)
==================================================

Implements the ESPCN architecture for real-time single-image super-resolution.

Key design principles for NVIDIA TensorRT deployment:
  - All feature extraction happens at LOW-RESOLUTION spatial dimensions.
  - Only the final PixelShuffle layer expands to high resolution.
  - No BatchNorm layers (TensorRT fuses Conv+ReLU but BN adds overhead and
    complicates FP16 calibration).
  - All convolutions use odd kernel sizes with symmetric padding for clean
    ONNX export and TensorRT optimization.
  - Weight initialization follows He (Kaiming) uniform for conv layers and
    ICNR (Iterative Closest Neighbour Resize) for the sub-pixel conv to avoid
    checkerboard artefacts at initialization.

Supported scale factors: 2, 3, 4.

Tensor shapes (example at scale=2, 1080p target):
  Input:  [B, 3, 540, 960]   (LR)
  Output: [B, 3, 1080, 1920] (HR)

All intermediate feature maps remain at 540×960 — this is why ESPCN is
fast enough for real-time: the expensive spatial computation is avoided.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init


def _icnr_init(tensor: torch.Tensor, scale_factor: int) -> None:
    """
    ICNR (Iterative Closest Neighbour Resize) initializer for sub-pixel
    convolution layers.  Prevents checkerboard artefacts by initializing
    the S² sub-kernels identically, producing a nearest-neighbour upscale
    at the start of training, then allowing the network to learn refinement.

    Parameters
    ----------
    tensor : torch.Tensor
        Weight tensor of shape [out_channels, in_channels, kH, kW] where
        out_channels = base_channels * scale_factor².
    scale_factor : int
        Upscaling factor (2, 3, or 4).
    """
    out_channels, in_channels, kh, kw = tensor.shape
    base_channels = out_channels // (scale_factor * scale_factor)

    # Initialize a base kernel with Kaiming uniform
    base_kernel = torch.empty(base_channels, in_channels, kh, kw)
    init.kaiming_uniform_(base_kernel, a=math.sqrt(5))

    # Tile the base kernel across the S² sub-pixel groups
    # Each group of S² output channels gets the same initial weights
    kernel = base_kernel.repeat(scale_factor * scale_factor, 1, 1, 1)
    tensor.data.copy_(kernel)


class ESPCN(nn.Module):
    """
    Efficient Sub-Pixel Convolutional Network.

    Architecture:
        Conv2d(3 → 64, 5×5)  →  ReLU
        Conv2d(64 → 64, 3×3) →  ReLU
        Conv2d(64 → 32, 3×3) →  ReLU
        Conv2d(32 → 3·S², 3×3)          ← sub-pixel conv (no activation)
        PixelShuffle(S)                   ← depth-to-space rearrangement

    All spatial operations are at LR resolution.  The final PixelShuffle
    converts channel depth to spatial resolution.

    Parameters
    ----------
    scale_factor : int
        Upscaling factor.  Must be 2, 3, or 4.
    num_channels : int
        Number of input/output color channels (default 3 for RGB).
    """

    SUPPORTED_SCALES = (2, 3, 4)

    def __init__(self, scale_factor: int = 2, num_channels: int = 3) -> None:
        super().__init__()
        if scale_factor not in self.SUPPORTED_SCALES:
            raise ValueError(
                f"scale_factor must be one of {self.SUPPORTED_SCALES}, "
                f"got {scale_factor}"
            )
        self.scale_factor = scale_factor
        self.num_channels = num_channels

        # ── Feature extraction at LR resolution ──────────────────────────
        # Layer 1: large receptive field to capture local structure
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=5,
            padding=2,  # same padding at LR
            bias=True,
        )
        # Layer 2: deepen feature representation
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        # Layer 3: compress channels before sub-pixel expansion
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # ── Sub-pixel upscaling layer ────────────────────────────────────
        # Outputs 3·S² channels at LR resolution, then PixelShuffle
        # rearranges to 3 channels at HR resolution.
        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=num_channels * (scale_factor ** 2),
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale_factor)

        # Non-linearity (no BN for TensorRT compatibility)
        self.relu = nn.ReLU(inplace=True)

        # ── Weight initialization ────────────────────────────────────────
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize weights:
        - Kaiming uniform for conv1, conv2, conv3 (matches ReLU activation)
        - ICNR for conv4 (sub-pixel layer) to prevent checkerboard artefacts
        - Zero-init all biases
        """
        for module in [self.conv1, self.conv2, self.conv3]:
            init.kaiming_uniform_(module.weight, a=0, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)

        # Sub-pixel conv: ICNR initialization
        _icnr_init(self.conv4.weight, self.scale_factor)
        if self.conv4.bias is not None:
            init.zeros_(self.conv4.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Low-resolution input of shape [B, 3, H, W].
            Expected range: [0, 1] (float32 or float16).

        Returns
        -------
        torch.Tensor
            High-resolution output of shape [B, 3, H*S, W*S].
            Output range: approximately [0, 1] (clamped).
        """
        # Feature extraction at LR
        out = self.relu(self.conv1(x))      # [B, 64, H, W]
        out = self.relu(self.conv2(out))     # [B, 64, H, W]
        out = self.relu(self.conv3(out))     # [B, 32, H, W]

        # Sub-pixel upscaling
        out = self.conv4(out)                # [B, 3*S², H, W]
        out = self.pixel_shuffle(out)        # [B, 3, H*S, W*S]

        # Clamp to valid range for image output
        out = torch.clamp(out, 0.0, 1.0)

        return out

    @torch.no_grad()
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_input_shape(self, target_h: int, target_w: int) -> tuple[int, int, int, int]:
        """
        Compute the required input shape for a desired output resolution.

        Parameters
        ----------
        target_h : int
            Target output height.
        target_w : int
            Target output width.

        Returns
        -------
        tuple[int, int, int, int]
            (batch=1, channels=3, height, width) for the LR input.
        """
        lr_h = target_h // self.scale_factor
        lr_w = target_w // self.scale_factor
        return (1, self.num_channels, lr_h, lr_w)


# ── Convenience constructors ─────────────────────────────────────────────

def espcn_2x(num_channels: int = 3) -> ESPCN:
    """Create an ESPCN model with 2× upscaling."""
    return ESPCN(scale_factor=2, num_channels=num_channels)


def espcn_3x(num_channels: int = 3) -> ESPCN:
    """Create an ESPCN model with 3× upscaling."""
    return ESPCN(scale_factor=3, num_channels=num_channels)


def espcn_4x(num_channels: int = 3) -> ESPCN:
    """Create an ESPCN model with 4× upscaling."""
    return ESPCN(scale_factor=4, num_channels=num_channels)


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for scale in ESPCN.SUPPORTED_SCALES:
        model = ESPCN(scale_factor=scale)
        params = model.count_parameters()
        # Simulate 1080p target output
        inp_shape = model.get_input_shape(1080, 1920)
        dummy = torch.randn(*inp_shape)
        out = model(dummy)
        print(
            f"ESPCN {scale}×  |  params: {params:,}  |  "
            f"input: {list(dummy.shape)}  →  output: {list(out.shape)}"
        )

        # Verify output shape
        expected_h = inp_shape[2] * scale
        expected_w = inp_shape[3] * scale
        assert out.shape == (1, 3, expected_h, expected_w), (
            f"Shape mismatch: expected (1, 3, {expected_h}, {expected_w}), "
            f"got {out.shape}"
        )
        # Verify output range
        assert out.min() >= 0.0 and out.max() <= 1.0, "Output out of [0, 1] range"

    print("\n✓ All ESPCN self-tests passed.")
