from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import torch
from OpenGL import GL
from pycuda.driver import memcpy_dtod_async

logger = logging.getLogger(__name__)


@dataclass
class CudaGlBuffer:
    pbo: int
    resource: object
    width: int
    height: int


class CudaGlInterop:
    """
    Register GL pixel unpack buffers and copy CUDA display tensors into them
    for sub-2ms texture upload via glTexSubImage2D offset=0.
    """

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._bytes = width * height * 4
        self._buffers: list[CudaGlBuffer] = []
        self._initialized = False

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def initialize(self) -> None:
        if self._initialized:
            return
        import pycuda.gl.autoinit  # noqa: F401
        import pycuda.gl as cudagl

        self._cudagl = cudagl
        for _ in range(2):
            pbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo)
            GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self._bytes, None, GL.GL_STREAM_DRAW)
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
            resource = cudagl.register_buffer(pbo, cudagl.graphics_map_flags.WRITE_DISCARD)
            self._buffers.append(CudaGlBuffer(pbo=pbo, resource=resource, width=self._width, height=self._height))

        self._initialized = True
        logger.info("CUDA-GL interop initialized %dx%d (%d bytes)", self._width, self._height, self._bytes)

    def resize(self, width: int, height: int) -> None:
        if width == self._width and height == self._height:
            return
        self.teardown()
        self._width = width
        self._height = height
        self._bytes = width * height * 4
        self.initialize()

    def teardown(self) -> None:
        for buf in self._buffers:
            try:
                buf.resource.unregister()
            except Exception:
                pass
            GL.glDeleteBuffers(1, [buf.pbo])
        self._buffers.clear()
        self._initialized = False

    def copy_tensor_to_pbo(self, rgba_tensor: torch.Tensor, index: int, stream: torch.cuda.Stream) -> int:
        if not self._initialized:
            self.initialize()
        if rgba_tensor.dtype != torch.uint8 or rgba_tensor.ndim != 3 or rgba_tensor.shape[2] != 4:
            raise ValueError(f"Expected RGBA uint8 HWC, got {tuple(rgba_tensor.shape)} {rgba_tensor.dtype}")
        buf = self._buffers[index & 1]
        mapping = buf.resource.map()
        try:
            ptr, size = mapping.device_ptr_and_size()
            nbytes = rgba_tensor.numel()
            if size < nbytes:
                raise RuntimeError(f"PBO too small: {size} < {nbytes}")
            memcpy_dtod_async(int(ptr), int(rgba_tensor.data_ptr()), nbytes, stream.cuda_stream)
        finally:
            mapping.unmap()
        return buf.pbo

    def upload_texture_from_pbo(self, texture_id: int, pbo: int) -> None:
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            self._width,
            self._height,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
