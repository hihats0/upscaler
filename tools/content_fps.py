"""Pencerenin gercek icerik kare hizi (sadece sayi, kayit yok).

WGC ile pencere yakalanir; her kare 1/8 kucultulup bir oncekiyle karsilastirilir. Icerik degistiyse
"yeni kare". Saniyede kac yeni kare, yeni kareler arasi sure dagilimi (20 ms = 50 FPS, 40 ms = kare
tekrari). Kareler bellekte kalir, diske SADECE sayi yazilir (TOD kurali).

    .venv/Scripts/python.exe tools/content_fps.py --window "upscaler" --seconds 20
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("tools", 1)[0])
from upscaler.winutil import dpi_aware, find_window  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--thr", type=float, default=0.3, help="ortalama mutlak fark esigi (0-255)")
    a = ap.parse_args()
    from windows_capture import WindowsCapture
    dpi_aware()
    hwnd = find_window(a.window)
    prev = [None]
    events: list[float] = []   # yeni icerik zamanlari
    arrivals = [0]
    done = threading.Event()
    cap = WindowsCapture(cursor_capture=False, draw_border=False, window_hwnd=hwnd, minimum_update_interval=1)

    @cap.event
    def on_frame_arrived(frame, control):
        if done.is_set():
            control.stop()
            return
        now = time.perf_counter()
        arrivals[0] += 1
        b = frame.frame_buffer
        h, w = b.shape[:2]
        small = b[int(h * .2):int(h * .8):8, int(w * .1):int(w * .9):8, 1].astype(np.int16)
        if prev[0] is not None and prev[0].shape == small.shape:
            if float(np.abs(small - prev[0]).mean()) > a.thr:
                events.append(now)
        prev[0] = small

    @cap.event
    def on_closed():
        pass

    ctl = cap.start_free_threaded()
    time.sleep(a.seconds)
    done.set()
    try:
        ctl.stop()
    except Exception:  # noqa: BLE001
        pass
    ev = np.array(events)
    d = np.diff(ev) * 1000 if len(ev) > 1 else np.array([0.0])
    span = ev[-1] - ev[0] if len(ev) > 1 else 1
    res = {"pencere": a.window, "yakalanan_olay": arrivals[0], "yeni_icerik": len(ev),
           "icerik_fps": round((len(ev) - 1) / span, 2) if len(ev) > 1 else 0,
           "aralik_ms_p5_p50_p95_max": [round(float(np.percentile(d, q)), 1) for q in (5, 50, 95, 100)],
           "aralik_dagilimi": {"<15ms": int((d < 15).sum()), "15-25ms (50fps)": int(((d >= 15) & (d < 25)).sum()),
                               "25-35ms": int(((d >= 25) & (d < 35)).sum()), "35-45ms (tekrar/25fps)": int(((d >= 35) & (d < 45)).sum()),
                               ">=45ms": int((d >= 45).sum())}}
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
