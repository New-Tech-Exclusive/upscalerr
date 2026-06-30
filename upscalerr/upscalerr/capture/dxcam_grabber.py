from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import dxcam
import torch

from upscalerr.capture.cuda_frame_view import CudaFrameView
from upscalerr.gpu.device_context import DeviceContext

logger = logging.getLogger(__name__)


@dataclass
class GrabState:
    frame_id: int = 0
    view: Optional[CudaFrameView] = None
    grab_result: object = None
    region: Optional[Tuple[int, int, int, int]] = None
    timestamp: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    event: threading.Event = field(default_factory=threading.Event)


class DxcamGrabber:
    """
    Dedicated GPU capture thread using dxcam BGRA output and zero-copy CUDA views.
    """

    def __init__(
        self,
        ctx: DeviceContext,
        monitor_index: int = 0,
        max_fps: int = 240,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self._ctx = ctx
        self._monitor_index = monitor_index
        self._max_fps = max_fps
        self._region = region
        self._camera = dxcam.create(
            device_idx=ctx.device_index,
            output_idx=monitor_index,
            output_color="BGRA",
        )
        self._state = GrabState()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._on_frame: Optional[Callable[[GrabState], None]] = None

    @property
    def region(self) -> Optional[Tuple[int, int, int, int]]:
        return self._region

    @region.setter
    def region(self, value: Optional[Tuple[int, int, int, int]]) -> None:
        self._region = value

    def set_frame_callback(self, cb: Callable[[GrabState], None]) -> None:
        self._on_frame = cb

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="DxcamGrabber", daemon=True)
        self._thread.start()
        logger.info("DXCam grab thread started (monitor=%s, region=%s)", self._monitor_index, self._region)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("DXCam grab thread stopped")

    def latest(self) -> GrabState:
        return self._state

    def _run(self) -> None:
        interval = 1.0 / max(self._max_fps, 1)
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick = max(next_tick + interval, time.perf_counter())

            with torch.cuda.stream(self._ctx.capture_stream):
                if self._region is not None:
                    left, top, right, bottom = self._region
                    grab = self._camera.grab(region=(left, top, right, bottom))
                else:
                    grab = self._camera.grab()

            if grab is None:
                continue

            try:
                view = CudaFrameView(grab, device=self._ctx.device)
            except RuntimeError as exc:
                logger.warning("Zero-copy view failed: %s", exc)
                continue

            with self._state.lock:
                self._state.frame_id += 1
                self._state.view = view
                self._state.grab_result = grab
                self._state.region = self._region
                self._state.timestamp = time.perf_counter()
                self._state.event.set()

            self._ctx.capture_done.record(self._ctx.capture_stream)

            if self._on_frame:
                self._on_frame(self._state)
