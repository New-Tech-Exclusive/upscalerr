from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import tensorrt as trt

logger = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


@dataclass
class TrtEngineBundle:
    engine_path: Path
    engine: trt.ICudaEngine
    context: trt.IExecutionContext
    io_names: List[str] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "TrtEngineBundle":
        engine_path = Path(path)
        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        runtime = trt.Runtime(TRT_LOGGER)
        blob = engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(blob)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(f"Failed to create execution context: {engine_path}")

        io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        input_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        output_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        bundle = cls(
            engine_path=engine_path,
            engine=engine,
            context=context,
            io_names=io_names,
            input_names=input_names,
            output_names=output_names,
        )
        logger.info(
            "Loaded TRT engine %s inputs=%s outputs=%s",
            engine_path.name,
            input_names,
            output_names,
        )
        return bundle

    def set_input_shape(self, name: str, shape: tuple[int, ...]) -> None:
        if not self.context.set_input_shape(name, shape):
            raise RuntimeError(f"Failed to set input shape for {name}: {shape}")

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        shape = self.context.get_tensor_shape(name)
        return tuple(int(x) for x in shape)

    def get_tensor_dtype(self, name: str) -> trt.DataType:
        return self.engine.get_tensor_dtype(name)
