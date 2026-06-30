from __future__ import annotations

import torch
import torch.nn.functional as F


def _make_base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)


def bilinear_warp(
    image: torch.Tensor,
    flow: torch.Tensor,
    dst: torch.Tensor,
    t: float = 0.5,
) -> torch.Tensor:
    """
    Backward bilinear warp at fractional time t along flow.

    image: [1,3,H,W] fp16/fp32
    flow:  [1,2,H,W] pixel displacements (dx, dy) from t=0 -> t=1
    dst:   preallocated [1,3,H,W]
    """
    if image.ndim != 4 or flow.ndim != 4:
        raise ValueError("image and flow must be NCHW")
    _, _, h, w = image.shape
    if flow.shape != (1, 2, h, w):
        raise ValueError(f"flow shape mismatch: {tuple(flow.shape)} vs image {(1,2,h,w)}")

    flow_t = flow.to(dtype=torch.float32) * float(t)
    base = _make_base_grid(h, w, device=image.device, dtype=torch.float32)

    dx = flow_t[:, 0]
    dy = flow_t[:, 1]
    px = (base[..., 0] + (dx / max(w - 1, 1)) * 2.0).clamp(-1.0, 1.0)
    py = (base[..., 1] + (dy / max(h - 1, 1)) * 2.0).clamp(-1.0, 1.0)
    grid = torch.stack((px, py), dim=-1)

    img_fp32 = image.to(dtype=torch.float32)
    warped = F.grid_sample(
        img_fp32,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    dst.copy_(warped.to(dtype=dst.dtype))
    return dst


def pyramidal_bilinear_warp(
    image: torch.Tensor,
    flow: torch.Tensor,
    dst: torch.Tensor,
    t: float = 0.5,
    levels: int = 2,
) -> torch.Tensor:
    """
    Two-level pyramidal warp: coarse pass at half resolution, refine at full res.
    """
    if levels <= 1:
        return bilinear_warp(image, flow, dst, t=t)

    _, _, h, w = image.shape
    h2, w2 = max(h // 2, 1), max(w // 2, 1)

    img_half = F.interpolate(image, size=(h2, w2), mode="bilinear", align_corners=True)
    flow_half = F.interpolate(flow, size=(h2, w2), mode="bilinear", align_corners=True) * 0.5

    coarse = torch.empty_like(img_half)
    bilinear_warp(img_half, flow_half, coarse, t=t)

    upsampled = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=True)
    residual_flow = flow - flow_half
    return bilinear_warp(upsampled, residual_flow, dst, t=t)
