from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time

import torch

from upscalerr.gpu.color_convert import bgra_uint8_hwc_to_rgb_fp16_nchw, rgb_fp16_nchw_to_rgba_uint8_hwc


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile color conversion kernels")
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda")
    bgra = torch.randint(0, 256, (args.height, args.width, 4), dtype=torch.uint8, device=device)
    rgb = torch.empty((1, 3, args.height, args.width), dtype=torch.float16, device=device)
    rgba = torch.empty((args.height, args.width, 4), dtype=torch.uint8, device=device)

    for _ in range(20):
        bgra_uint8_hwc_to_rgb_fp16_nchw(bgra, rgb)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        bgra_uint8_hwc_to_rgb_fp16_nchw(bgra, rgb)
        rgb_fp16_nchw_to_rgba_uint8_hwc(rgb, rgba)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / args.iters
    print(f"convert+pack avg {ms:.3f} ms @ {args.width}x{args.height}")


if __name__ == "__main__":
    main()
