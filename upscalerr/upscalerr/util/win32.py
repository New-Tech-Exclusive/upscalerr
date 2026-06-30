from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    rect: Tuple[int, int, int, int]  # left, top, right, bottom


def _is_visible_window(hwnd: int) -> bool:
    if not user32.IsWindowVisible(hwnd):
        return False
    if user32.GetWindowTextLengthW(hwnd) == 0:
        return False
    style = user32.GetWindowLongW(hwnd, -16)  # GWL_STYLE
    if style & 0x10000000 == 0:  # WS_VISIBLE
        return False
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return False
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    return width > 64 and height > 64


def enumerate_windows() -> List[WindowInfo]:
    windows: List[WindowInfo] = []

    def callback(hwnd: int, _lparam: int) -> bool:
        if not _is_visible_window(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd) + 1
        buffer = ctypes.create_unicode_buffer(length)
        user32.GetWindowTextW(hwnd, buffer, length)
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        windows.append(
            WindowInfo(
                hwnd=int(hwnd),
                title=buffer.value,
                rect=(rect.left, rect.top, rect.right, rect.bottom),
            )
        )
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    windows.sort(key=lambda w: w.title.lower())
    return windows


def get_foreground_window() -> Optional[WindowInfo]:
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    length = user32.GetWindowTextLengthW(hwnd) + 1
    buffer = ctypes.create_unicode_buffer(length)
    user32.GetWindowTextW(hwnd, buffer, length)
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return WindowInfo(
        hwnd=int(hwnd),
        title=buffer.value,
        rect=(rect.left, rect.top, rect.right, rect.bottom),
    )


def monitor_rect(monitor_index: int = 0) -> Tuple[int, int, int, int]:
    monitors: List[Tuple[int, int, int, int]] = []

    MonitorEnumProc = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )

    def cb(hmon, _hdc, lprect, _lparam):
        r = lprect.contents
        monitors.append((r.left, r.top, r.right, r.bottom))
        return True

    user32.EnumDisplayMonitors(0, 0, MonitorEnumProc(cb), 0)
    if not monitors:
        return (0, 0, 1920, 1080)
    idx = min(max(monitor_index, 0), len(monitors) - 1)
    return monitors[idx]


def window_region(hwnd: int) -> Tuple[int, int, int, int]:
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return (rect.left, rect.top, rect.right, rect.bottom)
