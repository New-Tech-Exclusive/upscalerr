import threading

import torch

from upscalerr.gpu.color_convert import bgra_uint8_hwc_to_rgb_fp16_nchw


def test_bgra_to_rgb_channel_order():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return
    bgra = torch.zeros((2, 2, 4), dtype=torch.uint8, device=device)
    bgra[..., 0] = 10  # B
    bgra[..., 1] = 20  # G
    bgra[..., 2] = 30  # R
    bgra[..., 3] = 255
    dst = torch.empty((1, 3, 2, 2), dtype=torch.float16, device=device)
    bgra_uint8_hwc_to_rgb_fp16_nchw(bgra, dst)
    assert abs(float(dst[0, 0, 0, 0]) - 30 / 255.0) < 1e-3
    assert abs(float(dst[0, 1, 0, 0]) - 20 / 255.0) < 1e-3
    assert abs(float(dst[0, 2, 0, 0]) - 10 / 255.0) < 1e-3
