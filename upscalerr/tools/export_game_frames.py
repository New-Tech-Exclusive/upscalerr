from __future__ import annotations

import argparse
import time
from pathlib import Path

import dxcam
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DXGI captures to PNG dataset")
    parser.add_argument("--output", default="data/frames/train")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--interval-ms", type=int, default=33)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cam = dxcam.create(output_color="BGRA")
    interval = args.interval_ms / 1000.0

    for i in range(args.count):
        frame = cam.grab()
        if frame is None:
            continue
        img = frame[..., :3][:, :, ::-1]
        Image.fromarray(img).save(out / f"frame_{i:05d}.png")
        time.sleep(interval)

    print(f"Saved {args.count} frames to {out}")


if __name__ == "__main__":
    main()
