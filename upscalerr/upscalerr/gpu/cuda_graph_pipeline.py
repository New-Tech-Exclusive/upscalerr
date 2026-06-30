from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from upscalerr.gpu.color_convert import (
    bgra_uint8_hwc_to_rgb_fp16_nchw,
    rgb_fp16_nchw_to_rgba_uint8_hwc,
)
from upscalerr.gpu.device_context import DeviceContext
from upscalerr.gpu.frame_buffer_pool import FrameBufferPool
from upscalerr.inference.espcn_engine import EspcnEngine
from upscalerr.inference.flownet_engine import FlowNetEngine
from upscalerr.inference.motion_warp import pyramidal_bilinear_warp

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutputs:
    display_real: torch.Tensor
    display_mid: Optional[torch.Tensor]
    has_mid: bool


class CudaGraphPipeline:
    """
    Pre-recorded CUDA graph for capture→convert→TRT→warp→display prep.
    Falls back to eager execution if graph capture is unsupported.
    """

    def __init__(
        self,
        ctx: DeviceContext,
        pool: FrameBufferPool,
        espcn: EspcnEngine,
        flownet: Optional[FlowNetEngine],
        enable_framegen: bool,
        flow_half_step: float = 0.5,
        pyramid_levels: int = 2,
    ) -> None:
        self._ctx = ctx
        self._pool = pool
        self._espcn = espcn
        self._flownet = flownet
        self._enable_framegen = enable_framegen and flownet is not None
        self._flow_t = flow_half_step
        self._pyramid_levels = pyramid_levels
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_bgra = torch.empty(
            (pool.height, pool.width, 4), dtype=torch.uint8, device=ctx.device
        )
        self._display_real_buf = pool.display_a
        self._display_mid_buf = pool.display_b
        self._has_prev = False
        self._use_graph = False

    def warmup_and_capture(self) -> None:
        stream = self._ctx.inference_stream
        for _ in range(3):
            self._run_eager(self._static_bgra, stream)

        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                self._run_eager(self._static_bgra, stream)
            self._graph = g
            self._use_graph = True
            logger.info("CUDA graph captured successfully")
        except RuntimeError as exc:
            logger.warning("CUDA graph capture failed, using eager path: %s", exc)
            self._graph = None
            self._use_graph = False

    def process_frame(self, bgra_view: torch.Tensor) -> PipelineOutputs:
        stream = self._ctx.inference_stream
        with torch.cuda.stream(stream):
            stream.wait_event(self._ctx.capture_done)
            if self._use_graph and self._graph is not None:
                self._static_bgra.copy_(bgra_view)
                self._graph.replay()
            else:
                self._run_eager(bgra_view, stream)
        self._ctx.inference_done.record(stream)

        has_mid = self._enable_framegen and self._has_prev
        return PipelineOutputs(
            display_real=self._display_real_buf,
            display_mid=self._display_mid_buf if has_mid else None,
            has_mid=has_mid,
        )

    def _run_eager(self, bgra: torch.Tensor, stream: torch.cuda.Stream) -> None:
        pool = self._pool
        bgra_uint8_hwc_to_rgb_fp16_nchw(bgra, pool.rgb_curr)
        self._espcn.run(pool)

        if self._enable_framegen and self._flownet is not None and self._has_prev:
            self._flownet.run(pool)
            flow_hr = self._flownet.upscale_flow(pool)
            assert pool.warp_mid is not None
            pyramidal_bilinear_warp(
                pool.espcn_out_prev,
                flow_hr,
                pool.warp_mid,
                t=self._flow_t,
                levels=self._pyramid_levels,
            )
            rgb_fp16_nchw_to_rgba_uint8_hwc(pool.warp_mid, self._display_mid_buf)
            rgb_fp16_nchw_to_rgba_uint8_hwc(pool.espcn_out, self._display_real_buf)
        else:
            rgb_fp16_nchw_to_rgba_uint8_hwc(pool.espcn_out, self._display_real_buf)

        if self._has_prev:
            pool.swap_temporal()
        else:
            pool.rgb_prev.copy_(pool.rgb_curr)
            pool.espcn_out_prev.copy_(pool.espcn_out)
            self._has_prev = True
