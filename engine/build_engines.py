"""
TensorRT Engine Builder — Compile ONNX Models to Optimized FP16 Engines
=========================================================================

Wraps NVIDIA's `trtexec` tool to compile ONNX models into TensorRT engine
files (.engine) with:
  - FP16 precision for Tensor Core utilization
  - Dynamic shape profiles (480p → 1080p → 4K input range)
  - FP16 I/O formats to avoid cast overhead at engine boundaries
  - Timing caches for fast rebuilds
  - Maximum builder optimization (level 5)

Usage:
    python build_engines.py --onnx-dir onnx \
                            --output-dir profiles \
                            --scale 2

Prerequisites:
    - TensorRT 10.x installed with trtexec on PATH
    - NVIDIA GPU with compute capability ≥ 7.0 (Turing+)

Engine files are GPU-specific: an engine built on an RTX 4060 will NOT
run on an RTX 3070.  Rebuild on the target hardware.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path


def find_trtexec() -> str:
    """
    Locate the trtexec executable.

    Search order:
      1. TRTEXEC_PATH environment variable
      2. System PATH
      3. Common TensorRT installation directories
    """
    # Check environment variable
    env_path = os.environ.get("TRTEXEC_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Check system PATH
    trtexec = shutil.which("trtexec")
    if trtexec:
        return trtexec

    # Check common installation paths (Windows)
    common_paths = [
        r"C:\Program Files\NVIDIA\TensorRT\bin\trtexec.exe",
        r"C:\TensorRT\bin\trtexec.exe",
    ]

    # Also check CUDA toolkit paths
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        common_paths.append(os.path.join(cuda_path, "bin", "trtexec.exe"))

    for path in common_paths:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "trtexec not found. Install TensorRT and ensure trtexec is on PATH, "
        "or set the TRTEXEC_PATH environment variable."
    )


def build_engine(
    trtexec_path: str,
    onnx_path: str,
    engine_path: str,
    timing_cache_path: str,
    input_name: str,
    min_shapes: str,
    opt_shapes: str,
    max_shapes: str,
    fp16: bool = True,
    builder_optimization_level: int = 5,
    verbose: bool = False,
) -> bool:
    """
    Build a TensorRT engine from an ONNX model using trtexec.

    Parameters
    ----------
    trtexec_path : str
        Path to trtexec executable.
    onnx_path : str
        Path to input ONNX model.
    engine_path : str
        Path for output .engine file.
    timing_cache_path : str
        Path for timing cache (reused across builds for speed).
    input_name : str
        Name of the input tensor in the ONNX graph.
    min_shapes : str
        Minimum dynamic shapes (e.g., "input:1x3x270x480").
    opt_shapes : str
        Optimal (most common) shapes for kernel autotuning.
    max_shapes : str
        Maximum supported shapes.
    fp16 : bool
        Enable FP16 precision.
    builder_optimization_level : int
        TensorRT builder optimization level (0-5). 5 = maximum.
    verbose : bool
        Print trtexec verbose output.

    Returns
    -------
    bool
        True if build succeeded.
    """
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--timingCacheFile={timing_cache_path}",
        f"--minShapes={input_name}:{min_shapes}",
        f"--optShapes={input_name}:{opt_shapes}",
        f"--maxShapes={input_name}:{max_shapes}",
        f"--builderOptimizationLevel={builder_optimization_level}",
    ]

    # fp16 flag removed as it's unsupported in TRT 11 trtexec

    if verbose:
        cmd.append("--verbose")

    # Add additional optimization flags
    cmd.extend([
        "--memPoolSize=workspace:4096M",  # 4GB workspace for kernel selection
        "--noTF32",          # Disable TF32 — we want pure FP16
    ])

    print(f"\n{'-'*60}")
    print(f"Building TensorRT engine:")
    print(f"  ONNX:   {onnx_path}")
    print(f"  Engine: {engine_path}")
    print(f"  FP16:   {fp16}")
    print(f"  Shapes: min={min_shapes}  opt={opt_shapes}  max={max_shapes}")
    print(f"  Optimization level: {builder_optimization_level}")
    print(f"{'-'*60}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=not verbose,
            text=True,
            timeout=3600,  # 1 hour timeout (level 5 optimization can be slow)
        )

        if not verbose and result.stdout:
            # Print key metrics from trtexec output
            for line in result.stdout.splitlines():
                if any(kw in line.lower() for kw in [
                    "gpu compute time", "latency", "throughput",
                    "total host", "enqueue", "h2d", "d2h",
                ]):
                    print(f"  {line.strip()}")

        engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"\n  OK Engine built successfully ({engine_size_mb:.1f} MB)")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n  X Engine build FAILED")
        if e.stderr:
            # Print last 20 lines of error output
            for line in e.stderr.splitlines()[-20:]:
                print(f"    {line}")
        return False

    except subprocess.TimeoutExpired:
        print(f"\n  ✗ Engine build TIMED OUT (>1 hour)")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TensorRT FP16 engines from ONNX models"
    )
    parser.add_argument(
        "--onnx-dir", type=str, default="onnx",
        help="Directory containing ONNX models"
    )
    parser.add_argument(
        "--output-dir", type=str, default="profiles",
        help="Output directory for .engine and .cache files"
    )
    parser.add_argument(
        "--scale", type=int, default=2, choices=[2, 3, 4],
        help="ESPCN scale factor (determines ONNX filename)"
    )
    parser.add_argument(
        "--opt-level", type=int, default=5, choices=range(6),
        help="Builder optimization level 0-5 (default: 5, slowest but fastest engines)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full trtexec output"
    )
    parser.add_argument(
        "--skip-espcn", action="store_true",
        help="Skip ESPCN engine build"
    )
    parser.add_argument(
        "--skip-flownet", action="store_true",
        help="Skip FlowNet engine build"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Locate trtexec
    try:
        trtexec = find_trtexec()
        print(f"Found trtexec: {trtexec}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    results: list[tuple[str, bool]] = []

    # ── ESPCN Engine ─────────────────────────────────────────────────────
    if not args.skip_espcn:
        espcn_onnx = os.path.join(args.onnx_dir, f"espcn_{args.scale}x.onnx")
        if not os.path.exists(espcn_onnx):
            print(f"⚠ ESPCN ONNX not found: {espcn_onnx}")
            results.append(("ESPCN", False))
        else:
            # Dynamic shape profiles for ESPCN
            # The backend passes the full captured frame as ESPCN input.
            # min:  480p source  → 480×854
            # opt: 1080p source  → 1080×1920
            # max:  4K source    → 2160×3840  (covers 2560×1440 and 4K)
            s = args.scale
            success = build_engine(
                trtexec_path=trtexec,
                onnx_path=espcn_onnx,
                engine_path=os.path.join(args.output_dir, f"espcn_{s}x_fp16.engine"),
                timing_cache_path=os.path.join(args.output_dir, "espcn_timing.cache"),
                input_name="input",
                min_shapes="1x3x270x480",     # ~480p source
                opt_shapes="1x3x1080x1920",   # 1080p source
                max_shapes="1x3x2160x3840",   # 4K source (covers 1440p/2560x1440)
                fp16=True,
                builder_optimization_level=args.opt_level,
                verbose=args.verbose,
            )
            results.append(("ESPCN", success))

    # ── FlowNet Engine ───────────────────────────────────────────────────
    if not args.skip_flownet:
        flownet_onnx = os.path.join(args.onnx_dir, "flownet.onnx")
        if not os.path.exists(flownet_onnx):
            print(f"⚠ FlowNet ONNX not found: {flownet_onnx}")
            results.append(("FlowNet", False))
        else:
            # FlowNet operates at full resolution (no downscaling input)
            # Dynamic shapes for various capture resolutions
            success = build_engine(
                trtexec_path=trtexec,
                onnx_path=flownet_onnx,
                engine_path=os.path.join(args.output_dir, "flownet_fp16.engine"),
                timing_cache_path=os.path.join(args.output_dir, "flownet_timing.cache"),
                input_name="frames_concat",
                min_shapes="1x6x270x480",     # ~480p source
                opt_shapes="1x6x1080x1920",   # 1080p source
                max_shapes="1x6x1440x2560",   # 1440p max (FlowNet at 4K is too heavy)
                fp16=True,
                builder_optimization_level=args.opt_level,
                verbose=args.verbose,
            )
            results.append(("FlowNet", success))

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Build Summary:")
    for name, success in results:
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")
    print(f"{'='*60}")

    all_success = all(s for _, s in results)
    if all_success:
        print(f"\nAll engines built. Output: {args.output_dir}/")
        print("Next step: Run the backend with these engine files.")
    else:
        print("\nSome builds failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
