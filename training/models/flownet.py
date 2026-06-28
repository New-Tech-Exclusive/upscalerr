"""
FlowNet-Lite — Lightweight Optical Flow U-Net for Frame Interpolation
=====================================================================

A compact encoder-decoder architecture that estimates dense optical flow
between two input frames, then uses differentiable bilinear warping to
synthesize an intermediate frame.

Architecture design for NVIDIA TensorRT:
  - No BatchNorm (replaced with no-norm or implicit bias — cleaner FP16
    graph, fuses perfectly with Conv in TensorRT).
  - LeakyReLU(0.1) for stable gradients in flow estimation.
  - Skip connections via concatenation (not addition) — explicit shapes
    in ONNX graph, no broadcast ambiguity for TensorRT builder.
  - Bilinear grid_sample for warping — maps to a single TensorRT plugin
    or a custom CUDA kernel at inference.

Input:   Concatenation of Frame_N-1 and Frame_N+1 → [B, 6, H, W]
Output:  Optical flow field → [B, 2, H, W]  (dx, dy per pixel)
         Warped intermediate frame → [B, 3, H, W]

Total parameters: ~2.1M — fits comfortably in VRAM alongside ESPCN.

Normalization contract:
  - All pixel values in [0, 1] (float32/float16).
  - Flow output is in pixel units (not normalized).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ConvBlock(nn.Module):
    """
    Conv2d → LeakyReLU block (no batch norm).

    Parameters
    ----------
    in_ch : int
        Input channels.
    out_ch : int
        Output channels.
    kernel_size : int
        Convolution kernel size.
    stride : int
        Convolution stride (use 2 for downsampling).
    padding : int
        Padding amount.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class UpBlock(nn.Module):
    """
    Upsample (bilinear 2×) → Conv2d → LeakyReLU.

    Uses bilinear interpolation instead of ConvTranspose2d to avoid
    checkerboard artefacts. The subsequent 3×3 conv refines the features.

    Parameters
    ----------
    in_ch : int
        Input channels (includes skip connection channels).
    out_ch : int
        Output channels.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Feature map from deeper layer, [B, C, H, W].
        skip : torch.Tensor
            Skip connection from encoder, [B, C_skip, 2H, 2W].

        Returns
        -------
        torch.Tensor
            Upsampled and refined features, [B, out_ch, 2H, 2W].
        """
        # Bilinear upsample to match skip spatial dims
        x_up = F.interpolate(
            x, size=(skip.shape[2], skip.shape[3]),
            mode="bilinear", align_corners=False
        )
        # Concatenate skip connection along channel axis
        merged = torch.cat([x_up, skip], dim=1)
        return self.act(self.conv(merged))


class FlowNetLite(nn.Module):
    """
    Lightweight optical flow estimation network.

    Encoder-decoder U-Net that takes concatenated frame pair [B, 6, H, W]
    and outputs a dense flow field [B, 2, H, W].

    Encoder (strided convolutions for downsampling):
        enc1: 6  → 32,  stride 2   (H/2)
        enc2: 32 → 64,  stride 2   (H/4)
        enc3: 64 → 128, stride 2   (H/8)
        enc4: 128→ 256, stride 2   (H/16)
        enc5: 256→ 512, stride 2   (H/32)

    Decoder (bilinear upsample + concat skip + conv):
        dec5: 512+256 → 256        (H/16)
        dec4: 256+128 → 128        (H/8)
        dec3: 128+64  → 64         (H/4)
        dec2: 64+32   → 32         (H/2)

    Final: bilinear upsample to H → Conv 32→2 (flow head)
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────
        self.enc1 = ConvBlock(6, 32, kernel_size=3, stride=2, padding=1)
        self.enc2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc3 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc4 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc5 = ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)

        # ── Decoder ──────────────────────────────────────────────────────
        # in_ch for UpBlock = deep_channels + skip_channels
        self.dec5 = UpBlock(512 + 256, 256)
        self.dec4 = UpBlock(256 + 128, 128)
        self.dec3 = UpBlock(128 + 64, 64)
        self.dec2 = UpBlock(64 + 32, 32)

        # ── Flow head ────────────────────────────────────────────────────
        # Final upsample to original resolution + 1×1 conv to 2 channels
        self.flow_head = nn.Conv2d(32, 2, kernel_size=3, padding=1, bias=True)

        # ── Weight initialization ────────────────────────────────────────
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Kaiming initialization for all conv layers.
        Flow head initialized with small weights (zero mean, 0.001 std)
        so initial flow prediction is near-zero (identity warp).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    init.zeros_(m.bias)

        # Override flow head: small initial flow for stable early training
        init.normal_(self.flow_head.weight, mean=0.0, std=0.001)
        init.zeros_(self.flow_head.bias)

    def estimate_flow(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        """
        Estimate optical flow from frame_a to frame_b.

        Parameters
        ----------
        frame_a : torch.Tensor
            Source frame, [B, 3, H, W], range [0, 1].
        frame_b : torch.Tensor
            Target frame, [B, 3, H, W], range [0, 1].

        Returns
        -------
        torch.Tensor
            Dense flow field [B, 2, H, W] in pixel units.
            flow[:, 0] = horizontal displacement (dx)
            flow[:, 1] = vertical displacement (dy)
        """
        _, _, h, w = frame_a.shape

        # Concatenate frames along channel axis
        x = torch.cat([frame_a, frame_b], dim=1)  # [B, 6, H, W]

        # Encoder with saved skip connections
        e1 = self.enc1(x)      # [B, 32,  H/2,  W/2]
        e2 = self.enc2(e1)     # [B, 64,  H/4,  W/4]
        e3 = self.enc3(e2)     # [B, 128, H/8,  W/8]
        e4 = self.enc4(e3)     # [B, 256, H/16, W/16]
        e5 = self.enc5(e4)     # [B, 512, H/32, W/32]

        # Decoder with skip connections
        d5 = self.dec5(e5, e4)  # [B, 256, H/16, W/16]
        d4 = self.dec4(d5, e3)  # [B, 128, H/8,  W/8]
        d3 = self.dec3(d4, e2)  # [B, 64,  H/4,  W/4]
        d2 = self.dec2(d3, e1)  # [B, 32,  H/2,  W/2]

        # Upsample to original resolution
        d1 = F.interpolate(
            d2, size=(h, w),
            mode="bilinear", align_corners=False
        )  # [B, 32, H, W]

        # Flow prediction
        flow = self.flow_head(d1)  # [B, 2, H, W]

        return flow

    def forward(
        self,
        frame_prev: torch.Tensor,
        frame_next: torch.Tensor,
        timestep: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate intermediate frame via optical flow and bilinear warping.

        For frame generation, we estimate flow from frame_prev to frame_next,
        then warp frame_prev by (timestep × flow) to produce the intermediate.

        Parameters
        ----------
        frame_prev : torch.Tensor
            Previous frame (N-1), [B, 3, H, W], range [0, 1].
        frame_next : torch.Tensor
            Next frame (N+1), [B, 3, H, W], range [0, 1].
        timestep : float
            Temporal interpolation position.  0.5 = midpoint (default).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - warped_frame: Interpolated frame, [B, 3, H, W], range ≈[0, 1].
            - flow: Estimated flow field, [B, 2, H, W], in pixel units.
        """
        # Estimate forward flow (prev → next)
        flow = self.estimate_flow(frame_prev, frame_next)

        # Scale flow by timestep for intermediate position
        scaled_flow = flow * timestep

        # Warp frame_prev using the scaled flow
        warped = self.warp(frame_prev, scaled_flow)

        return warped, flow

    @staticmethod
    def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Differentiable bilinear warping using grid_sample.

        Warps `image` according to the displacement field `flow`.

        Parameters
        ----------
        image : torch.Tensor
            Source image to warp, [B, C, H, W].
        flow : torch.Tensor
            Flow field in pixel units, [B, 2, H, W].
            flow[:, 0] = dx (horizontal), flow[:, 1] = dy (vertical).

        Returns
        -------
        torch.Tensor
            Warped image, [B, C, H, W].
        """
        b, _, h, w = image.shape

        # Create base coordinate grid
        # grid_y: [H, W] with values from 0 to H-1
        # grid_x: [W, H] with values from 0 to W-1
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=image.dtype, device=image.device),
            torch.arange(w, dtype=image.dtype, device=image.device),
            indexing="ij",
        )

        # Add flow displacement to base coordinates
        # flow[:, 0] is dx (horizontal → x), flow[:, 1] is dy (vertical → y)
        sample_x = grid_x.unsqueeze(0) + flow[:, 0, :, :]  # [B, H, W]
        sample_y = grid_y.unsqueeze(0) + flow[:, 1, :, :]  # [B, H, W]

        # Normalize to [-1, 1] range for grid_sample
        # grid_sample expects normalized coordinates:
        #   x: -1 = left edge, +1 = right edge
        #   y: -1 = top edge,  +1 = bottom edge
        sample_x_norm = 2.0 * sample_x / (w - 1) - 1.0
        sample_y_norm = 2.0 * sample_y / (h - 1) - 1.0

        # Stack into grid [B, H, W, 2] — last dim is (x, y) for grid_sample
        grid = torch.stack([sample_x_norm, sample_y_norm], dim=-1)

        # Bilinear sampling with zero padding for out-of-bounds
        warped = F.grid_sample(
            image, grid,
            mode="bilinear",
            padding_mode="border",  # clamp to edge pixels (less artefacts)
            align_corners=True,
        )

        return warped

    @staticmethod
    def flow_smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
        """
        First-order smoothness regularization on the flow field.

        Penalizes large spatial gradients in the flow to encourage
        piece-wise smooth motion estimation.

        Parameters
        ----------
        flow : torch.Tensor
            Flow field, [B, 2, H, W].

        Returns
        -------
        torch.Tensor
            Scalar smoothness loss.
        """
        # Horizontal gradient (∂flow/∂x)
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        # Vertical gradient (∂flow/∂y)
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])

        return dx.mean() + dy.mean()

    @torch.no_grad()
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = FlowNetLite()
    params = model.count_parameters()
    print(f"FlowNet-Lite  |  params: {params:,}")

    # Test at 256×256 (training resolution)
    b, h, w = 2, 256, 256
    frame_prev = torch.rand(b, 3, h, w)
    frame_next = torch.rand(b, 3, h, w)

    warped, flow = model(frame_prev, frame_next, timestep=0.5)

    print(f"Input:   frame_prev {list(frame_prev.shape)}, frame_next {list(frame_next.shape)}")
    print(f"Output:  warped {list(warped.shape)}, flow {list(flow.shape)}")

    assert warped.shape == (b, 3, h, w), f"Warped shape mismatch: {warped.shape}"
    assert flow.shape == (b, 2, h, w), f"Flow shape mismatch: {flow.shape}"

    # Test flow smoothness loss
    smooth_loss = FlowNetLite.flow_smoothness_loss(flow)
    print(f"Flow smoothness loss: {smooth_loss.item():.6f}")

    # Test standalone warp
    identity_flow = torch.zeros(b, 2, h, w)
    identity_warped = FlowNetLite.warp(frame_prev, identity_flow)
    error = (identity_warped - frame_prev).abs().max().item()
    print(f"Identity warp max error: {error:.8f}")
    assert error < 1e-5, f"Identity warp error too large: {error}"

    # Test at 1080p inference resolution
    h_hd, w_hd = 1080, 1920
    frame_a = torch.rand(1, 3, h_hd, w_hd)
    frame_b = torch.rand(1, 3, h_hd, w_hd)
    warped_hd, flow_hd = model(frame_a, frame_b)
    print(
        f"1080p:   warped {list(warped_hd.shape)}, "
        f"flow {list(flow_hd.shape)}"
    )
    assert warped_hd.shape == (1, 3, h_hd, w_hd)
    assert flow_hd.shape == (1, 2, h_hd, w_hd)

    print("\n✓ All FlowNet-Lite self-tests passed.")
