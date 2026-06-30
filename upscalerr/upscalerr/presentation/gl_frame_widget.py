from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import torch
from OpenGL import GL
from PySide6.QtCore import Qt, QTimer
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat

from upscalerr.gpu.cuda_graph_pipeline import PipelineOutputs
from upscalerr.gpu.device_context import DeviceContext
from upscalerr.pipeline.frame_state import FrameState, PresentSlot
from upscalerr.presentation.cuda_gl_interop import CudaGlInterop
from upscalerr.util.timing import FrameTimer, LatencyTracker

logger = logging.getLogger(__name__)

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTex;
out vec2 vTex;
void main() {
    vTex = aTex;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 vTex;
out vec4 FragColor;
uniform sampler2D uFrame;
void main() {
    FragColor = texture(uFrame, vTex);
}
"""


def _compile_shader(source: str, shader_type: int) -> int:
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not status:
        log = GL.glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile failed: {log}")
    return shader


def _link_program(vs: int, fs: int) -> int:
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vs)
    GL.glAttachShader(program, fs)
    GL.glLinkProgram(program)
    status = GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
    if not status:
        log = GL.glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program link failed: {log}")
    return program


class GLFrameWidget(QOpenGLWidget):
    def __init__(
        self,
        ctx: DeviceContext,
        frame_state: FrameState,
        width: int,
        height: int,
        target_fps: float = 120.0,
        parent=None,
    ) -> None:
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setVersion(3, 3)
        fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
        fmt.setSwapInterval(0)
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__(parent)
        self._ctx = ctx
        self._frame_state = frame_state
        self._target_fps = target_fps
        self._out_w = width
        self._out_h = height
        self._program = 0
        self._vao = 0
        self._vbo = 0
        self._texture = 0
        self._interop: Optional[CudaGlInterop] = None
        self._outputs_lock = threading.Lock()
        self._latest_outputs: Optional[PipelineOutputs] = None
        self._pbo_index = 0
        self._present_latency = LatencyTracker()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)

    def set_output_size(self, width: int, height: int) -> None:
        self._out_w = width
        self._out_h = height
        if self._interop:
            self._interop.resize(width, height)

    def submit_outputs(self, outputs: PipelineOutputs) -> None:
        with self._outputs_lock:
            self._latest_outputs = outputs

    def start_present_loop(self) -> None:
        interval_ms = max(int(1000.0 / self._target_fps), 1)
        self._timer.start(interval_ms)

    def stop_present_loop(self) -> None:
        self._timer.stop()

    def initializeGL(self) -> None:
        GL.glClearColor(0.0, 0.0, 0.0, 0.0)
        vs = _compile_shader(VERTEX_SHADER, GL.GL_VERTEX_SHADER)
        fs = _compile_shader(FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER)
        self._program = _link_program(vs, fs)
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)

        vertices = np.array(
            [
                -1.0, -1.0, 0.0, 1.0,
                 1.0, -1.0, 1.0, 1.0,
                -1.0,  1.0, 0.0, 0.0,
                 1.0,  1.0, 1.0, 0.0,
            ],
            dtype=np.float32,
        )
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, GL.ctypes.c_void_p(8))

        self._texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8,
            self._out_w, self._out_h, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None,
        )

        self._interop = CudaGlInterop(self._out_w, self._out_h)
        self._interop.initialize()

    def _pick_rgba(self) -> Optional[torch.Tensor]:
        with self._outputs_lock:
            outputs = self._latest_outputs
        if outputs is None:
            return None
        slot = self._frame_state.pop_present()
        if slot == PresentSlot.MID and outputs.display_mid is not None:
            return outputs.display_mid
        return outputs.display_real

    def paintGL(self) -> None:
        timer = FrameTimer()
        GL.glViewport(0, 0, self.width(), self.height())
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        rgba = self._pick_rgba()
        if rgba is None:
            return

        with torch.cuda.stream(self._ctx.present_stream):
            self._ctx.present_stream.wait_event(self._ctx.inference_done)
            pbo = self._interop.copy_tensor_to_pbo(rgba, self._pbo_index, self._ctx.present_stream)
            self._pbo_index += 1

        self._interop.upload_texture_from_pbo(self._texture, pbo)

        GL.glUseProgram(self._program)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glUniform1i(GL.glGetUniformLocation(self._program, "uFrame"), 0)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        self._present_latency.record_ms(timer.elapsed_ms())
        self._frame_state.present_fps = 1000.0 / max(self._present_latency.avg_ms, 0.001)

    def resizeGL(self, w: int, h: int) -> None:
        GL.glViewport(0, 0, w, h)

    def cleanup_gl(self) -> None:
        if self._interop:
            self._interop.teardown()
        if self._texture:
            GL.glDeleteTextures(1, [self._texture])
        if self._vbo:
            GL.glDeleteBuffers(1, [self._vbo])
        if self._vao:
            GL.glDeleteVertexArrays(1, [self._vao])
        if self._program:
            GL.glDeleteProgram(self._program)
