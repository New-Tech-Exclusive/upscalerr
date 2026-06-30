from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


@dataclass
class LatencyTracker:
    window_size: int = 120
    _samples: Deque[float] = field(default_factory=lambda: deque(maxlen=120))

    def record_ms(self, value: float) -> None:
        self._samples.append(value)

    @property
    def avg_ms(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    @property
    def p95_ms(self) -> float:
        if not self._samples:
            return 0.0
        ordered = sorted(self._samples)
        idx = int(0.95 * (len(ordered) - 1))
        return ordered[idx]


class FrameTimer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> float:
        now = time.perf_counter()
        dt = (now - self._t0) * 1000.0
        self._t0 = now
        return dt


class RateLimiter:
    def __init__(self, target_fps: float) -> None:
        self._interval = 1.0 / max(target_fps, 1.0)
        self._next = time.perf_counter()

    def wait(self) -> None:
        now = time.perf_counter()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(self._next + self._interval, time.perf_counter())
