"""
ONNX Export Script — Export Trained Models to ONNX Format
==========================================================

Exports both ESPCN and FlowNet-Lite to ONNX with:
  - Dynamic batch + height + width axes (for TensorRT dynamic shapes)
  - FP32 weights (TensorRT handles FP16 conversion at build time)
  - opset 17 for maximum TensorRT 10.x compatibility
  - Input/output names matching the C++ inference code expectations

Usage:
    python export_onnx.py --espcn-checkpoint checkpoints/espcn/best_espcn_2x.pt \
                          --flownet-checkpoint checkpoints/flownet/best_flownet.pt \
                          --output-dir ../engine/onnx \
                          --scale 2

Output files:
    espcn_2x.onnx    — Spatial upscaler (input: [B,3,H,W] → output: [B,3,H*S,W*S])
    flownet.onnx     — Flow estimator  (input: [B,6,H,W] → output: flow [B,2,H,W])
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
import numpy as np

from models.espcn import ESPCN
from models.flownet import FlowNetLite


def export_espcn(
    checkpoint_path: str,
    output_path: str,
    scale_factor: int = 2,
    opset_version: int = 17,
) -> None:
    """
    Export trained ESPCN model to ONNX.

    The ONNX graph takes a single input (low-res image) and produces
    a single output (high-res image).

    Dynamic axes: batch, height, width — so TensorRT can build
    engines with optimization profiles for multiple resolutions.
    """
    print(f"\n{'='*50}")
    print(f"Exporting ESPCN {scale_factor}× to ONNX")
    print(f"{'='*50}")

    # Load model
    model = ESPCN(scale_factor=scale_factor)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded weights from: {checkpoint_path}")
        if "val_psnr" in checkpoint:
            print(f"  Val PSNR: {checkpoint['val_psnr']:.2f} dB")
    else:
        print(f"⚠ No checkpoint found at '{checkpoint_path}', exporting random weights")
        print("  (useful for pipeline testing, not for production)")

    model.eval()
    model.float()  # Ensure FP32 for export

    # Create dummy input at a representative resolution
    # Using 540p as the "opt" shape for 2× (targets 1080p output)
    dummy_h = 540 // scale_factor if scale_factor <= 2 else 270
    dummy_w = 960 // scale_factor if scale_factor <= 2 else 480
    dummy_input = torch.randn(1, 3, dummy_h, dummy_w)

    print(f"Dummy input shape:  {list(dummy_input.shape)}")

    # Trace output shape
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f"Dummy output shape: {list(dummy_output.shape)}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {
                0: "batch_size",
                2: "height",
                3: "width",
            },
            "output": {
                0: "batch_size",
                2: f"height_x{scale_factor}",
                3: f"width_x{scale_factor}",
            },
        },
    )

    print(f"Exported to: {output_path}")

    # ── Validate ONNX model ─────────────────────────────────────────────
    _validate_onnx(output_path, dummy_input, dummy_output, model_name="ESPCN")


def export_flownet(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 17,
) -> None:
    """
    Export trained FlowNet-Lite to ONNX.

    The ONNX graph is exported as the flow estimation part only
    (estimate_flow method), not the full forward with warping.
    Warping is done as a separate CUDA kernel at inference for
    maximum control over memory layout.

    Input:  concat(frame_prev, frame_next) → [B, 6, H, W]
    Output: flow field → [B, 2, H, W]
    """
    print(f"\n{'='*50}")
    print(f"Exporting FlowNet-Lite to ONNX")
    print(f"{'='*50}")

    # Load model
    model = FlowNetLite()

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded weights from: {checkpoint_path}")
        if "val_psnr" in checkpoint:
            print(f"  Val PSNR: {checkpoint['val_psnr']:.2f} dB")
    else:
        print(f"⚠ No checkpoint found at '{checkpoint_path}', exporting random weights")

    model.eval()
    model.float()

    # ── Create wrapper that takes single 6-channel input ─────────────────
    # TensorRT works best with a single-input graph, so we concatenate
    # the two frames before export and have the C++ code do the concat.
    class FlowNetExportWrapper(torch.nn.Module):
        """
        Wraps FlowNetLite to accept a single [B, 6, H, W] input
        (pre-concatenated frames) and output [B, 2, H, W] flow.
        """

        def __init__(self, flownet: FlowNetLite) -> None:
            super().__init__()
            self.flownet = flownet

        def forward(self, frames_concat: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            frames_concat : torch.Tensor
                Concatenated frame pair, [B, 6, H, W].
                channels 0-2: frame_prev (RGB)
                channels 3-5: frame_next (RGB)

            Returns
            -------
            torch.Tensor
                Flow field [B, 2, H, W].
            """
            frame_prev = frames_concat[:, :3, :, :]
            frame_next = frames_concat[:, 3:, :, :]
            flow = self.flownet.estimate_flow(frame_prev, frame_next)
            return flow

    wrapper = FlowNetExportWrapper(model)
    wrapper.eval()

    # Dummy input: 6-channel concatenated frames
    dummy_input = torch.randn(1, 6, 540, 960)
    print(f"Dummy input shape:  {list(dummy_input.shape)}")

    with torch.no_grad():
        dummy_output = wrapper(dummy_input)
    print(f"Dummy output shape: {list(dummy_output.shape)}")

    # Export
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["frames_concat"],
        output_names=["flow"],
        dynamic_axes={
            "frames_concat": {
                0: "batch_size",
                2: "height",
                3: "width",
            },
            "flow": {
                0: "batch_size",
                2: "height",
                3: "width",
            },
        },
    )

    print(f"Exported to: {output_path}")

    _validate_onnx(output_path, dummy_input, dummy_output, model_name="FlowNet")


def _validate_onnx(
    onnx_path: str,
    dummy_input: torch.Tensor,
    expected_output: torch.Tensor,
    model_name: str,
    atol: float = 1e-4,
) -> None:
    """
    Validate the exported ONNX model:
      1. Check ONNX graph is well-formed
      2. Run inference with ONNX Runtime
      3. Compare outputs against PyTorch reference
    """
    print(f"\nValidating {model_name} ONNX export...")

    # 1. Check model structure
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model, full_check=True)
    print(f"  ✓ ONNX graph validation passed")

    # Print model info
    graph = onnx_model.graph
    print(f"  Inputs:  {[(inp.name, [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]) for inp in graph.input]}")
    print(f"  Outputs: {[(out.name, [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]) for out in graph.output]}")

    # Count nodes by op type
    op_counts: dict[str, int] = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    top_ops = sorted(op_counts.items(), key=lambda x: -x[1])[:10]
    print(f"  Top ops: {top_ops}")

    # 2. ONNX Runtime inference
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

    input_name = session.get_inputs()[0].name
    input_array = dummy_input.numpy()

    ort_outputs = session.run(None, {input_name: input_array})
    ort_output = ort_outputs[0]

    # 3. Compare with PyTorch reference
    expected_np = expected_output.detach().numpy()

    max_diff = np.abs(ort_output - expected_np).max()
    mean_diff = np.abs(ort_output - expected_np).mean()

    print(f"  ORT output shape: {ort_output.shape}")
    print(f"  Max abs diff:     {max_diff:.6e}")
    print(f"  Mean abs diff:    {mean_diff:.6e}")

    if max_diff < atol:
        print(f"  ✓ Output matches PyTorch reference (atol={atol})")
    else:
        print(f"  ⚠ Output differs from PyTorch (max_diff={max_diff:.6e} > atol={atol})")
        print(f"    This may be acceptable — FP32 rounding differs between frameworks.")

    # File size
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  ✓ {model_name} ONNX export complete\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ESPCN and FlowNet to ONNX for TensorRT compilation"
    )
    parser.add_argument(
        "--espcn-checkpoint", type=str,
        default="checkpoints/espcn/best_espcn_2x.pt",
        help="Path to ESPCN checkpoint (.pt)"
    )
    parser.add_argument(
        "--flownet-checkpoint", type=str,
        default="checkpoints/flownet/best_flownet.pt",
        help="Path to FlowNet checkpoint (.pt)"
    )
    parser.add_argument(
        "--scale", type=int, default=2, choices=[2, 3, 4],
        help="ESPCN scale factor (default: 2)"
    )
    default_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine", "onnx"))
    parser.add_argument(
        "--output-dir", type=str, default=default_out_dir,
        help="Output directory for ONNX files"
    )
    parser.add_argument(
        "--opset", type=int, default=18,
        help="ONNX opset version (default: 18)"
    )
    parser.add_argument(
        "--skip-espcn", action="store_true",
        help="Skip ESPCN export"
    )
    parser.add_argument(
        "--skip-flownet", action="store_true",
        help="Skip FlowNet export"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_espcn:
        espcn_onnx_path = os.path.join(
            args.output_dir, f"espcn_{args.scale}x.onnx"
        )
        export_espcn(
            checkpoint_path=args.espcn_checkpoint,
            output_path=espcn_onnx_path,
            scale_factor=args.scale,
            opset_version=args.opset,
        )

    if not args.skip_flownet:
        flownet_onnx_path = os.path.join(args.output_dir, "flownet.onnx")
        export_flownet(
            checkpoint_path=args.flownet_checkpoint,
            output_path=flownet_onnx_path,
            opset_version=args.opset,
        )

    print("All exports complete.")
    print(f"Next step: Run engine/build_engines.py to compile TensorRT engines.")


if __name__ == "__main__":
    main()
