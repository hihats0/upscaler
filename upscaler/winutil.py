"""Windows pencere yardimcilari."""
from __future__ import annotations

import ctypes

import win32api
import win32con
import win32gui


def dpi_aware() -> None:
    """Gercek piksel koordinatlari (Windows olcekleme %125 olsa bile)."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        pass


def fine_timer() -> None:
    """time.sleep hassasiyetini 1 ms'e indirir."""
    ctypes.windll.winmm.timeBeginPeriod(1)


def find_window(title: str) -> int:
    """Basliginda title gecen ilk gorunur pencere."""
    hits: list[int] = []

    def cb(h, _):
        if win32gui.IsWindowVisible(h) and title in win32gui.GetWindowText(h):
            hits.append(h)
    win32gui.EnumWindows(cb, None)
    if not hits:
        raise SystemExit(f"pencere bulunamadi: {title!r}")
    return hits[0]


def bring_front(hwnd: int) -> None:
    win32api.keybd_event(0x12, 0, 0, 0)  # ALT: SetForegroundWindow kisitini asmak icin
    win32gui.SetForegroundWindow(hwnd)
    win32api.keybd_event(0x12, 0, win32con.KEYEVENTF_KEYUP, 0)
