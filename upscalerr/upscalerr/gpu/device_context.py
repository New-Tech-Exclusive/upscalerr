from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class DeviceContext:
    device_index: int = 0

    def __post_init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available")
        self.device = torch.device(f"cuda:{self.device_index}")
        torch.cuda.set_device(self.device)
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.inference_stream = torch.cuda.Stream(device=self.device)
        self.present_stream = torch.cuda.Stream(device=self.device)
        self.capture_done = torch.cuda.Event(enable_timing=True)
        self.inference_done = torch.cuda.Event(enable_timing=True)
        self._pool: List[torch.cuda.Stream] = [
            torch.cuda.Stream(device=self.device) for _ in range(4)
        ]
        self._pool_idx = 0

    def borrow_stream(self) -> torch.cuda.Stream:
        stream = self._pool[self._pool_idx]
        self._pool_idx = (self._pool_idx + 1) % len(self._pool)
        return stream

    def synchronize_device(self) -> None:
        torch.cuda.synchronize(self.device)

    def vram_used_mb(self) -> float:
        return torch.cuda.memory_allocated(self.device) / (1024.0 * 1024.0)

    def vram_reserved_mb(self) -> float:
        return torch.cuda.memory_reserved(self.device) / (1024.0 * 1024.0)
