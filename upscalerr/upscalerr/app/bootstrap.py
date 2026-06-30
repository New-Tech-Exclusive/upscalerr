from __future__ import annotations

import upscalerr.util.env  # noqa: F401 — KMP_DUPLICATE_LIB_OK on Windows

import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from PySide6.QtWidgets import QApplication

from upscalerr.capture.dxcam_grabber import DxcamGrabber
from upscalerr.capture.window_target import WindowTarget
from upscalerr.gpu.cuda_graph_pipeline import CudaGraphPipeline, PipelineOutputs
from upscalerr.gpu.device_context import DeviceContext
from upscalerr.gpu.frame_buffer_pool import FrameBufferPool
from upscalerr.inference.espcn_engine import EspcnEngine
from upscalerr.inference.flownet_engine import FlowNetEngine
from upscalerr.pipeline.frame_state import FrameState
from upscalerr.pipeline.inference_worker import InferenceWorker
from upscalerr.presentation.overlay_window import OverlayWindow
from upscalerr.ui.main_window import MainWindow

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class UpscalerrApp:
    def __init__(self, config_path: str | Path) -> None:
        self._config = self._load_config(config_path)
        self._ctx = DeviceContext(
            device_index=int(self._config["device"]["cuda_device_index"])
        )
        self._frame_state = FrameState()
        self._target: Optional[WindowTarget] = None
        self._grabber: Optional[DxcamGrabber] = None
        self._worker: Optional[InferenceWorker] = None
        self._overlay: Optional[OverlayWindow] = None
        self._pipeline: Optional[CudaGraphPipeline] = None
        self._pool: Optional[FrameBufferPool] = None
        self._running = False

    @staticmethod
    def _load_config(path: str | Path) -> Dict[str, Any]:
        cfg_path = Path(path)
        if not cfg_path.is_absolute():
            candidate = PROJECT_ROOT / cfg_path
            if candidate.is_file():
                cfg_path = candidate
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_engine_path(self, key: str) -> Path:
        raw = self._config["engines"][key]
        p = Path(raw)
        if p.is_absolute():
            return p
        for base in (PROJECT_ROOT, Path.cwd()):
            candidate = base / p
            if candidate.is_file():
                return candidate
        return PROJECT_ROOT / p

    def engine_paths(self) -> Dict[str, Path]:
        return {
            "espcn": self._resolve_engine_path("espcn_path"),
            "flownet": self._resolve_engine_path("flownet_path"),
        }

    def engine_ready(self) -> bool:
        return self.engine_paths()["espcn"].is_file()

    def engine_warnings(self) -> list[str]:
        warnings: list[str] = []
        paths = self.engine_paths()
        if not paths["espcn"].is_file():
            warnings.append(f"Missing ESPCN engine: {paths['espcn']}")
        if bool(self._config["pipeline"]["frame_generation"]) and not paths["flownet"].is_file():
            warnings.append(f"Frame generation is enabled but FlowNet engine is missing: {paths['flownet']}")
        return warnings

    def _build_pipeline(self, width: int, height: int) -> None:
        scale = int(self._config["engines"]["scale"])
        framegen = bool(self._config["pipeline"]["frame_generation"])
        espcn_path = self._resolve_engine_path("espcn_path")
        flownet_path = self._resolve_engine_path("flownet_path")

        self._pool = FrameBufferPool(
            device=self._ctx.device,
            height=height,
            width=width,
            scale=scale,
            enable_framegen=framegen,
        )
        espcn = EspcnEngine(espcn_path, self._ctx.inference_stream)
        espcn.configure(self._pool)

        flownet = None
        if framegen and flownet_path.is_file():
            flownet = FlowNetEngine(flownet_path, self._ctx.inference_stream, scale=scale)
            flownet.configure(self._pool)
        elif framegen:
            logger.warning("Frame generation enabled but FlowNet engine missing: %s", flownet_path)

        self._pipeline = CudaGraphPipeline(
            ctx=self._ctx,
            pool=self._pool,
            espcn=espcn,
            flownet=flownet,
            enable_framegen=framegen and flownet is not None,
            flow_half_step=float(self._config["pipeline"]["flow_half_step"]),
            pyramid_levels=int(self._config["pipeline"]["pyramid_levels"]),
        )
        self._pipeline.warmup_and_capture()

    def set_frame_generation(self, enabled: bool) -> None:
        self._config["pipeline"]["frame_generation"] = enabled

    def set_capture_fps(self, max_fps: int) -> None:
        self._config["capture"]["max_fps"] = int(max_fps)

    def set_target_present_fps(self, target_fps: int) -> None:
        self._config["pipeline"]["target_present_fps"] = int(target_fps)

    def set_monitor_index(self, monitor_index: int) -> None:
        self._config["overlay"]["monitor_index"] = int(monitor_index)

    def set_transparent_input(self, enabled: bool) -> None:
        self._config["overlay"]["transparent_input"] = enabled

    def start(self, target: WindowTarget) -> None:
        if self._running:
            self.stop()

        region = target.capture_region()
        if region is None:
            raise RuntimeError("Invalid capture target window")

        left, top, right, bottom = region
        width = right - left
        height = bottom - top
        if width < 64 or height < 64:
            raise RuntimeError(f"Capture region too small: {width}x{height}")

        self._target = target
        self._build_pipeline(width, height)

        scale = int(self._config["engines"]["scale"])
        out_w, out_h = width * scale, height * scale

        self._overlay = OverlayWindow(
            ctx=self._ctx,
            frame_state=self._frame_state,
            monitor_index=int(self._config["overlay"]["monitor_index"]),
            output_size=(out_w, out_h),
            target_fps=float(self._config["pipeline"]["target_present_fps"]),
            transparent_input=bool(self._config["overlay"]["transparent_input"]),
        )

        self._grabber = DxcamGrabber(
            ctx=self._ctx,
            monitor_index=int(self._config["overlay"]["monitor_index"]),
            max_fps=int(self._config["capture"]["max_fps"]),
            region=region,
        )
        self._grabber.start()

        assert self._pipeline is not None
        self._worker = InferenceWorker(
            ctx=self._ctx,
            pipeline=self._pipeline,
            frame_state=self._frame_state,
            on_outputs=self._on_pipeline_outputs,
        )
        self._worker.attach_grabber(self._grabber.latest())
        self._worker.start()
        self._overlay.show_overlay()
        self._running = True
        logger.info("Upscaler running on %s (%dx%d -> %dx%d)", target.title, width, height, out_w, out_h)

    def stop(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker = None
        if self._grabber:
            self._grabber.stop()
            self._grabber = None
        if self._overlay:
            self._overlay.hide_overlay()
            self._overlay = None
        self._pipeline = None
        self._pool = None
        self._running = False
        logger.info("Upscaler stopped")

    @property
    def running(self) -> bool:
        return self._running

    @property
    def frame_state(self) -> FrameState:
        return self._frame_state

    @property
    def device_context(self) -> DeviceContext:
        return self._ctx

    def _on_pipeline_outputs(self, outputs: PipelineOutputs) -> None:
        if self._overlay is None:
            return
        self._overlay.submit_frame(outputs)


def run_gui(config_path: str | Path) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Upscalerr")
    app.setOrganizationName("Upscalerr")
    app.setStyle("Fusion")

    style_path = Path(__file__).resolve().parents[1] / "ui" / "resources" / "style.qss"
    if style_path.is_file():
        try:
            app.setStyleSheet(style_path.read_text(encoding="utf-8"))
        except OSError as exc:
            logger.warning("Failed to load stylesheet %s: %s", style_path, exc)

    core = UpscalerrApp(config_path)
    window = MainWindow(core)
    window.show()

    def _shutdown(*_args):
        core.stop()
        app.quit()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    return app.exec()
