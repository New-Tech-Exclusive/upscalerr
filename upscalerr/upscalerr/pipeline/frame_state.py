from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class PresentSlot(Enum):
    REAL = auto()
    MID = auto()


@dataclass
class FrameState:
    frame_id: int = 0
    present_queue: list[PresentSlot] = field(default_factory=list)
    capture_fps: float = 0.0
    present_fps: float = 0.0
    inference_ms: float = 0.0

    def enqueue_real(self) -> None:
        self.present_queue.append(PresentSlot.REAL)

    def enqueue_mid_then_real(self) -> None:
        self.present_queue.append(PresentSlot.MID)
        self.present_queue.append(PresentSlot.REAL)

    def pop_present(self) -> PresentSlot | None:
        if not self.present_queue:
            return None
        return self.present_queue.pop(0)
