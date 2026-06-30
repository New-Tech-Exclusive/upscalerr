from __future__ import annotations

import logging
from typing import Dict

import tensorrt as trt
import torch

from upscalerr.inference.trt_runtime import TrtEngineBundle

logger = logging.getLogger(__name__)


class TrtExecutor:
    """Bind PyTorch tensor device pointers directly to TensorRT IO tensors."""

    def __init__(self, bundle: TrtEngineBundle, stream: torch.cuda.Stream) -> None:
        self._bundle = bundle
        self._stream = stream
        self._bindings: Dict[str, torch.Tensor] = {}

    @property
    def context(self) -> trt.IExecutionContext:
        return self._bundle.context

    @property
    def input_names(self) -> list[str]:
        return self._bundle.input_names

    @property
    def output_names(self) -> list[str]:
        return self._bundle.output_names

    def configure_shapes(self, shapes: Dict[str, tuple[int, ...]]) -> None:
        for name, shape in shapes.items():
            if name in self._bundle.input_names:
                self._bundle.set_input_shape(name, shape)

    def bind(self, tensors: Dict[str, torch.Tensor]) -> None:
        ctx = self._bundle.context
        for name, tensor in tensors.items():
            if not tensor.is_cuda:
                raise ValueError(f"Tensor {name} must be CUDA")
            if not tensor.is_contiguous():
                raise ValueError(f"Tensor {name} must be contiguous for TRT binding")
            ok = ctx.set_tensor_address(name, int(tensor.data_ptr()))
            if not ok:
                raise RuntimeError(f"set_tensor_address failed for {name}")
            self._bindings[name] = tensor

    def execute(self) -> None:
        stream_handle = self._stream.cuda_stream
        ok = self._bundle.context.execute_async_v3(stream_handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

    def execute_bound(self, tensors: Dict[str, torch.Tensor]) -> None:
        self.bind(tensors)
        self.execute()
