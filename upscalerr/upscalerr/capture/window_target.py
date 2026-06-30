from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from upscalerr.util.win32 import WindowInfo, window_region


@dataclass
class WindowTarget:
    hwnd: Optional[int] = None
    title: str = ""

    @classmethod
    def from_window(cls, info: WindowInfo) -> "WindowTarget":
        return cls(hwnd=info.hwnd, title=info.title)

    def capture_region(self) -> Optional[Tuple[int, int, int, int]]:
        if self.hwnd is None:
            return None
        return window_region(self.hwnd)

    def is_valid(self) -> bool:
        return self.hwnd is not None
