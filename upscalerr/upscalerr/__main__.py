from __future__ import annotations

import argparse
from pathlib import Path

import upscalerr.util.env  # noqa: F401

from upscalerr.app.bootstrap import run_gui


def main() -> int:
    parser = argparse.ArgumentParser(description="Upscalerr - pure Python game upscaler")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"),
        help="Path to YAML config",
    )
    args = parser.parse_args()
    return run_gui(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
