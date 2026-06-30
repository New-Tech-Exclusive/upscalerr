from __future__ import annotations

from upscalerr.gpu.cuda_graph_pipeline import PipelineOutputs
from upscalerr.pipeline.frame_state import FrameState, PresentSlot


class FrameGenSequencer:
    """
    Schedules interleaved presentation: Frame N -> N-0.5 (warped) -> N+1.
    """

    def __init__(self, frame_state: FrameState) -> None:
        self._state = frame_state
        self._pending_real: PipelineOutputs | None = None
        self._pending_mid: PipelineOutputs | None = None

    def on_inference(self, outputs: PipelineOutputs) -> None:
        self._pending_real = outputs

    def next_display_tensor(self, outputs: PipelineOutputs):
        slot = self._state.pop_present()
        if slot is None:
            return outputs.display_real
        if slot == PresentSlot.MID and outputs.display_mid is not None:
            return outputs.display_mid
        return outputs.display_real
