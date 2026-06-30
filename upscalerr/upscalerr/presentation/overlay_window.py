from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from upscalerr.gpu.device_context import DeviceContext
from upscalerr.pipeline.frame_state import FrameState
from upscalerr.presentation.gl_frame_widget import GLFrameWidget
from upscalerr.util.win32 import monitor_rect


class OverlayWindow(QWidget):
    """Borderless fullscreen overlay for presenting upscaled frames."""

    def __init__(
        self,
        ctx: DeviceContext,
        frame_state: FrameState,
        monitor_index: int = 0,
        output_size: Tuple[int, int] = (2560, 1440),
        target_fps: float = 120.0,
        transparent_input: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._monitor_index = monitor_index
        left, top, right, bottom = monitor_rect(monitor_index)
        self.setGeometry(left, top, right - left, bottom - top)

        flags = Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        if transparent_input:
            flags |= Qt.WindowTransparentForInput
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

        ow, oh = output_size[0], output_size[1]
        self.gl_widget = GLFrameWidget(ctx, frame_state, ow, oh, target_fps=target_fps, parent=self)
        self.gl_widget.setGeometry(0, 0, self.width(), self.height())

    def show_overlay(self) -> None:
        self.showFullScreen()
        self.gl_widget.start_present_loop()

    def hide_overlay(self) -> None:
        self.gl_widget.stop_present_loop()
        self.hide()

    def submit_frame(self, outputs) -> None:
        self.gl_widget.submit_outputs(outputs)

    def resizeEvent(self, event) -> None:
        self.gl_widget.setGeometry(0, 0, self.width(), self.height())
        super().resizeEvent(event)

    def closeEvent(self, event) -> None:
        self.gl_widget.cleanup_gl()
        super().closeEvent(event)
