"""WGC (Windows.Graphics.Capture) kaynagi.

Onemli (olculdu, 2026-09-15): MinUpdateInterval varsayilani ~16 ms. 144 Hz
ekranda bu, yakalamayi 48 Hz'e kisip 50 FPS kaynagi katlar (her 25 karede bir
kare kaybolur). Bu yuzden varsayilan 1 ms.

Geri cagirmaya gelen tampon SADECE cagri icinde gecerlidir; tuketici hemen kopyalar.
"""
from __future__ import annotations

import threading
from typing import Callable

import numpy as np
from windows_capture import WindowsCapture

# on_new_frame(raw_zaman_sn, bgra_gorunumu)
FrameCallback = Callable[[float, np.ndarray], None]


class WgcSource:
    def __init__(self, hwnd: int, on_new_frame: FrameCallback, min_interval_ms: int = 1) -> None:
        self.hwnd = hwnd
        self.on_new_frame = on_new_frame
        self.min_interval_ms = min_interval_ms
        self.delivered = 0   # WGC'nin verdigi tum kareler
        self.unique = 0      # icerigi degisen kareler
        self.size = (0, 0)
        self.first_raw: float | None = None
        self.last_raw: float | None = None
        self.error: BaseException | None = None
        self._prev: np.ndarray | None = None
        self._stop = threading.Event()
        self._control = None

    def start(self) -> "WgcSource":
        cap = WindowsCapture(cursor_capture=False, draw_border=False, window_hwnd=self.hwnd,
                             minimum_update_interval=self.min_interval_ms)

        @cap.event
        def on_frame_arrived(frame, control):
            if self._stop.is_set():
                control.stop()
                return
            try:
                self._handle(frame)
            except BaseException as e:  # geri cagri icinde hata sessizce kaybolmasin
                self.error = e
                control.stop()

        @cap.event
        def on_closed():
            pass

        self._control = cap.start_free_threaded()
        return self

    def _handle(self, frame) -> None:
        buf = frame.frame_buffer
        self.delivered += 1
        self.size = (frame.width, frame.height)
        small = buf[::8, ::8, 1]
        if self._prev is not None and small.shape == self._prev.shape and np.array_equal(small, self._prev):
            return  # ayni kare (vsync tekrari)
        self._prev = small.copy()
        self.unique += 1
        raw = frame.timespan / 1e7
        if self.first_raw is None:
            self.first_raw = raw
        self.last_raw = raw
        self.on_new_frame(raw, buf)

    def unique_fps(self) -> float | None:
        """Ilk ve son benzersiz kare arasindaki gercek kare hizi."""
        if self.unique < 2 or self.first_raw is None or self.last_raw == self.first_raw:
            return None
        return (self.unique - 1) / (self.last_raw - self.first_raw)

    def stop(self) -> None:
        self._stop.set()
        if self._control is not None:
            self._control.stop()
