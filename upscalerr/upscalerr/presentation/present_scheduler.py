from __future__ import annotations

from upscalerr.util.timing import RateLimiter


class PresentScheduler:
    def __init__(self, target_fps: float = 120.0) -> None:
        self._limiter = RateLimiter(target_fps)

    def pace(self) -> None:
        self._limiter.wait()
