from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from upscalerr.gpu.frame_buffer_pool import FrameBufferPool
from upscalerr.inference.trt_executor import TrtExecutor
from upscalerr.inference.trt_runtime import TrtEngineBundle


class EspcnEngine:
    INPUT_NAME = "input"
    OUTPUT_NAME = "output"

    def __init__(self, engine_path: str | Path, stream: torch.cuda.Stream) -> None:
        self._bundle = TrtEngineBundle.load(engine_path)
        self._executor = TrtExecutor(self._bundle, stream)
        in_name = self._bundle.input_names[0]
        out_name = self._bundle.output_names[0]
        self.input_name = in_name
        self.output_name = out_name

    def configure(self, pool: FrameBufferPool) -> None:
        h, w = pool.height, pool.width
        oh, ow = pool.output_height, pool.output_width
        self._executor.configure_shapes({self.input_name: (1, 3, h, w)})
        expected_out = self._executor.context.get_tensor_shape(self.output_name)
        if tuple(expected_out) != (1, 3, oh, ow):
            raise RuntimeError(
                f"ESPCN output shape mismatch: engine={expected_out} pool={(1,3,oh,ow)}"
            )

    def run(self, pool: FrameBufferPool) -> torch.Tensor:
        tensors: Dict[str, torch.Tensor] = {
            self.input_name: pool.rgb_curr,
            self.output_name: pool.espcn_out,
        }
        self._executor.execute_bound(tensors)
        return pool.espcn_out
