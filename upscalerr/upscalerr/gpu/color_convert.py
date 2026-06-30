from __future__ import annotations

import torch


def bgra_uint8_hwc_to_rgb_fp16_nchw(
    bgra: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    """
    Convert DXGI BGRA uint8 [H,W,4] view into normalized RGB fp16 [1,3,H,W].

    Channel mapping (explicit, no implicit assumptions):
      B = bgra[..., 0]
      G = bgra[..., 1]
      R = bgra[..., 2]
      A = bgra[..., 3]  (ignored for ML input)

    dst must be preallocated [1,3,H,W] float16 on the same CUDA device.
    """
    if bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError(f"Expected BGRA HWC, got shape {tuple(bgra.shape)}")
    if bgra.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 BGRA, got {bgra.dtype}")
    h, w, _ = bgra.shape
    if dst.shape != (1, 3, h, w):
        raise ValueError(f"dst shape mismatch: expected (1,3,{h},{w}), got {tuple(dst.shape)}")
    if dst.dtype != torch.float16:
        raise ValueError("dst must be float16")

    inv255 = torch.tensor(1.0 / 255.0, device=bgra.device, dtype=torch.float32)
    b = bgra[..., 0].to(dtype=torch.float32).mul_(inv255)
    g = bgra[..., 1].to(dtype=torch.float32).mul_(inv255)
    r = bgra[..., 2].to(dtype=torch.float32).mul_(inv255)

    dst[0, 0].copy_(r.to(torch.float16))
    dst[0, 1].copy_(g.to(torch.float16))
    dst[0, 2].copy_(b.to(torch.float16))
    return dst


def rgb_fp16_nchw_to_rgba_uint8_hwc(
    rgb: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    """
    Convert normalized RGB fp16 [1,3,H,W] or [3,H,W] into RGBA uint8 [H,W,4] for GL upload.
    Alpha is set to 255.
    """
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB, got {tuple(rgb.shape)}")
    h, w = rgb.shape[1], rgb.shape[2]
    if dst.shape != (h, w, 4):
        raise ValueError(f"dst shape mismatch: expected ({h},{w},4), got {tuple(dst.shape)}")
    if dst.dtype != torch.uint8:
        raise ValueError("dst must be uint8")

    clamped = rgb.clamp(0.0, 1.0).to(torch.float32)
    scaled = (clamped * 255.0).to(torch.uint8)
    dst[..., 0] = scaled[0]  # R
    dst[..., 1] = scaled[1]  # G
    dst[..., 2] = scaled[2]  # B
    dst[..., 3] = 255
    return dst


def concat_rgb_pair_nchw(prev_rgb: torch.Tensor, curr_rgb: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Stack two [1,3,H,W] fp16 tensors into [1,6,H,W] for FlowNet input."""
    if prev_rgb.shape != curr_rgb.shape:
        raise ValueError("prev/curr RGB shape mismatch")
    if dst.shape != (1, 6, prev_rgb.shape[2], prev_rgb.shape[3]):
        raise ValueError(f"dst shape mismatch for flow concat: {tuple(dst.shape)}")
    dst[:, 0:3].copy_(prev_rgb)
    dst[:, 3:6].copy_(curr_rgb)
    return dst
