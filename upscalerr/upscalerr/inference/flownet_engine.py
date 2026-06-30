from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from upscalerr.gpu.frame_buffer_pool import FrameBufferPool
from upscalerr.inference.trt_executor import TrtExecutor
from upscalerr.inference.trt_runtime import TrtEngineBundle


class FlowNetEngine:
    INPUT_NAME = "input"
    OUTPUT_NAME = "output"

    def __init__(self, engine_path: str | Path, stream: torch.cuda.Stream, scale: int = 2) -> None:
        self._bundle = TrtEngineBundle.load(engine_path)
        self._executor = TrtExecutor(self._bundle, stream)
        self._scale = scale
        in_name = self._bundle.input_names[0]
        out_name = self._bundle.output_names[0]
        self.input_name = in_name
        self.output_name = out_name

    def configure(self, pool: FrameBufferPool) -> None:
        h, w = pool.height, pool.width
        self._executor.configure_shapes({self.input_name: (1, 6, h, w)})
        expected_out = self._executor.context.get_tensor_shape(self.output_name)
        if tuple(expected_out) != (1, 2, h, w):
            raise RuntimeError(
                f"FlowNet output shape mismatch: engine={expected_out} pool={(1,2,h,w)}"
            )

    def run(self, pool: FrameBufferPool) -> torch.Tensor:
        if pool.flow_in is None or pool.flow_out is None:
            raise RuntimeError("Frame generation buffers not allocated")
        from upscalerr.gpu.color_convert import concat_rgb_pair_nchw

        concat_rgb_pair_nchw(pool.rgb_prev, pool.rgb_curr, pool.flow_in)
        tensors: Dict[str, torch.Tensor] = {
            self.input_name: pool.flow_in,
            self.output_name: pool.flow_out,
        }
        self._executor.execute_bound(tensors)
        return pool.flow_out

    def upscale_flow(self, pool: FrameBufferPool) -> torch.Tensor:
        """Bilinear upsample flow field from capture resolution to ESPCN output resolution."""
        if pool.flow_out is None or pool.flow_out_hr is None:
            raise RuntimeError("Flow HR buffer missing")
        flow = pool.flow_out
        oh, ow = pool.output_height, pool.output_width
        scaled = F.interpolate(flow, size=(oh, ow), mode="bilinear", align_corners=True)
        scaled = scaled * float(self._scale)
        pool.flow_out_hr.copy_(scaled)
        return pool.flow_out_hr
