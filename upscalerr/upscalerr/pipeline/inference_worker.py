from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import torch

from upscalerr.capture.dxcam_grabber import GrabState
from upscalerr.gpu.cuda_graph_pipeline import CudaGraphPipeline, PipelineOutputs
from upscalerr.gpu.device_context import DeviceContext
from upscalerr.pipeline.frame_state import FrameState, PresentSlot
from upscalerr.util.timing import FrameTimer, LatencyTracker

logger = logging.getLogger(__name__)


class InferenceWorker:
    def __init__(
        self,
        ctx: DeviceContext,
        pipeline: CudaGraphPipeline,
        frame_state: FrameState,
        on_outputs: Callable[[PipelineOutputs], None],
    ) -> None:
        self._ctx = ctx
        self._pipeline = pipeline
        self._frame_state = frame_state
        self._on_outputs = on_outputs
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._grab_state: Optional[GrabState] = None
        self._latency = LatencyTracker()
        self._timer = FrameTimer()
        self._last_frame_id = -1

    @property
    def latency(self) -> LatencyTracker:
        return self._latency

    def attach_grabber(self, state: GrabState) -> None:
        self._grab_state = state

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="InferenceWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        if self._grab_state is None:
            raise RuntimeError("Grabber state not attached")

        while not self._stop.is_set():
            if not self._grab_state.event.wait(timeout=0.05):
                continue
            self._grab_state.event.clear()

            with self._grab_state.lock:
                view = self._grab_state.view
                frame_id = self._grab_state.frame_id
                if view is None or frame_id == self._last_frame_id:
                    continue
                self._last_frame_id = frame_id
                bgra = view.tensor

            timer = FrameTimer()
            outputs = self._pipeline.process_frame(bgra)
            self._latency.record_ms(timer.elapsed_ms())
            self._frame_state.frame_id = frame_id
            self._frame_state.inference_ms = self._latency.avg_ms

            if outputs.has_mid:
                self._frame_state.enqueue_mid_then_real()
            else:
                self._frame_state.enqueue_real()

            self._on_outputs(outputs)
