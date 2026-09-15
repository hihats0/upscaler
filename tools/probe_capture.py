"""WGC yakalama olcumu (Hat 1.0 / 1.1 testi).

Bir pencereyi Windows.Graphics.Capture ile birkac saniye yakalar ve SADECE
istatistik basar: kare boyutu, gelen/benzersiz kare hizi, aralik dagilimi,
orta bolge parlakligi, WGC zaman damgasi ile perf_counter iliskisi.
Test deseni icin (--pattern) barkod okunur: dusen kare ve ideal zaman hatasi.
Kareler bellekte kalir, diske yazilmaz.

Kullanim:
    .venv/Scripts/python.exe tools/probe_capture.py --title "TOD - Google Chrome" --seconds 5
    .venv/Scripts/python.exe tools/probe_capture.py --title upscaler-test-pattern --pattern
"""
from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import win32gui  # noqa: E402
from windows_capture import WindowsCapture  # noqa: E402

from upscaler.testpattern import decode  # noqa: E402
from upscaler.timeline import SourceClock  # noqa: E402
from upscaler.winutil import bring_front, dpi_aware, find_window  # noqa: E402


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def probe(hwnd: int, seconds: float, pattern: bool, min_interval: int | None) -> dict:
    arrivals: list[float] = []      # her gelen kare (perf_counter)
    spans: list[tuple[float, float]] = []
    new_times: list[float] = []     # benzersiz kareler
    unique_raw: list[float] = []    # benzersiz karelerin WGC zamani (tekrar oynatma testi icin)
    decoded_all: list[int] = []     # her benzersiz kare icin barkod (-1 = okunamadi)
    stats: list[tuple[float, float, float]] = []
    decoded: list[tuple[int, float]] = []
    size = [0, 0]
    prev = [None]
    done = threading.Event()

    cap = WindowsCapture(cursor_capture=False, draw_border=False, window_hwnd=hwnd,
                         minimum_update_interval=min_interval)

    @cap.event
    def on_frame_arrived(frame, control):
        now = time.perf_counter()
        buf = frame.frame_buffer
        arrivals.append(now)
        spans.append((frame.timespan / 1e7, now))
        size[0], size[1] = frame.width, frame.height
        small = buf[::8, ::8, 1].copy()
        if prev[0] is None or small.shape != prev[0].shape or not np.array_equal(small, prev[0]):
            new_times.append(now)
            unique_raw.append(frame.timespan / 1e7)
            idx = decode(buf) if pattern else None
            decoded_all.append(-1 if idx is None else idx)
            if idx is not None:
                decoded.append((idx, frame.timespan / 1e7))
            if len(new_times) % 10 == 1:
                h, w = small.shape
                c = small[h // 4:3 * h // 4, w // 4:3 * w // 4].astype(np.float32)
                stats.append((float(c.mean()), float(c.std()), float((c < 16).mean())))
        prev[0] = small
        if done.is_set():
            control.stop()

    @cap.event
    def on_closed():
        pass

    ctl = cap.start_free_threaded()
    time.sleep(seconds)
    done.set()
    t_end = time.perf_counter() + 2.0
    while not ctl.is_finished() and time.perf_counter() < t_end:
        time.sleep(0.02)
    if not ctl.is_finished():
        ctl.stop()

    out: dict = {"boyut": f"{size[0]}x{size[1]}", "gelen_kare": len(arrivals),
                 "_timing": list(zip(unique_raw, decoded_all))}
    if len(new_times) > 2:
        span = new_times[-1] - new_times[0]
        iv = np.diff(new_times) * 1000
        out.update({
            "benzersiz_fps": round((len(new_times) - 1) / span, 2),
            "gelen_fps": round((len(arrivals) - 1) / (arrivals[-1] - arrivals[0]), 2),
            "aralik_ms_p10_p50_p90": [round(pct(iv, 10), 1), round(pct(iv, 50), 1), round(pct(iv, 90), 1)],
        })
    if stats:
        out["orta_parlaklik_ort"] = round(statistics.mean(s[0] for s in stats), 1)
        out["orta_std_ort"] = round(statistics.mean(s[1] for s in stats), 1)
        out["orta_siyah_orani"] = round(statistics.mean(s[2] for s in stats), 3)
    if spans:
        offs = [p - s for s, p in spans]
        out["wgc_zaman_ofseti_ms_p50_p90"] = [round(pct(offs, 50) * 1000, 2), round(pct(offs, 90) * 1000, 2)]
    if pattern and len(decoded) > 2:
        idxs = [i for i, _ in decoded]
        expected = idxs[-1] - idxs[0] + 1
        clock = SourceClock()
        st = [clock.push(t) for _, t in decoded]
        # Saat, desenin kendi numarasini dogru tahmin ediyor mu?
        base = st[0].index - idxs[0]
        idx_err = sum(1 for s, i in zip(st, idxs) if s.index - base != i)
        tail = st[len(st) // 3:]
        gaps = [b.t - a.t - (clock.period or 0) * (b.index - a.index) for a, b in zip(tail, tail[1:])]
        out.update({
            "desen_okunan": len(decoded),
            "desen_beklenen": expected,
            "desen_dusen_kare": expected - len(set(idxs)),
            "saat_periyot_ms": round((clock.period or 0) * 1000, 3),
            "saat_sira_hatasi": idx_err,
            "ideal_aralik_hatasi_ms_max": round(max(abs(g) for g in gaps) * 1000, 3) if gaps else None,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--foreground", action="store_true", help="olcum sirasinda pencereyi one al")
    ap.add_argument("--pattern", action="store_true", help="test deseni barkodunu oku")
    ap.add_argument("--min-interval", type=int, default=1,
                    help="WGC MinUpdateInterval (ms). 0 = sistem varsayilani")
    ap.add_argument("--dump", help="benzersiz kare zamanlarini CSV'ye yaz (sadece sayi: raw_sn, barkod)")
    args = ap.parse_args()

    dpi_aware()
    hwnd = find_window(args.title)
    prev_fg = win32gui.GetForegroundWindow()
    try:
        if args.foreground:
            bring_front(hwnd)
            time.sleep(0.8)
        result = probe(hwnd, args.seconds, args.pattern, args.min_interval or None)
    finally:
        if args.foreground and prev_fg:
            try:
                bring_front(prev_fg)
            except Exception:
                pass
    timing = result.pop("_timing")
    if args.dump:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
        with open(args.dump, "w", encoding="utf-8") as f:
            f.write("raw_s,barcode\n")
            for raw, code in timing:
                f.write(f"{raw:.7f},{code}\n")
        result["dump"] = f"{args.dump} ({len(timing)} satir)"
    result["min_interval_ms"] = args.min_interval or "varsayilan"
    result["pencere"] = win32gui.GetWindowText(hwnd)
    result["on_planda_miydi"] = args.foreground
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
