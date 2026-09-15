"""Uygulama sesi yakalama olcumu (Hat 1.1 ses).

Chrome'un sesini WASAPI process loopback (ProcTap, surec agaci dahil) ile
yakalar. Ses SADECE bellekte olculur (RMS, parca boyutu, gelis araligi),
diske yazilmaz.

--mute-test: Asil soru. Chrome'u Windows karistiricisinda sessize alinca
yakalama da sessizlesiyor mu? Sessizlesmiyorsa: orijinal sesi kapatip
gecikmeli sesi biz calabiliriz. Test sonunda sessize alma geri acilir.

Kullanim:
    .venv/Scripts/python.exe tools/probe_audio.py --seconds 4
    .venv/Scripts/python.exe tools/probe_audio.py --mute-test
"""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import psutil
from proctap import ProcessAudioCapture


def chrome_main_pid() -> int:
    """Ebeveyni chrome.exe olmayan chrome.exe = tarayici ana sureci."""
    procs = [p for p in psutil.process_iter(["name", "ppid", "pid"]) if (p.info["name"] or "").lower() == "chrome.exe"]
    pids = {p.info["pid"] for p in procs}
    roots = [p.info["pid"] for p in procs if p.info["ppid"] not in pids]
    if not roots:
        raise SystemExit("chrome.exe bulunamadi")
    return roots[0]


class Meter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.chunks: list[tuple[float, int, float]] = []  # (gelis, kare sayisi, rms)

    def on_data(self, pcm: bytes, frames: int) -> None:
        now = time.perf_counter()
        x = np.frombuffer(pcm, dtype=np.float32)
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        with self.lock:
            self.chunks.append((now, x.size // 2, rms))

    def window(self, t0: float, t1: float) -> dict:
        with self.lock:
            c = [ch for ch in self.chunks if t0 <= ch[0] < t1]
        if not c:
            return {"parca": 0}
        frames = sum(f for _, f, _ in c)
        rms = float(np.sqrt(np.average([r * r for _, _, r in c], weights=[max(f, 1) for _, f, _ in c])))
        iv = np.diff([t for t, _, _ in c]) * 1000 if len(c) > 1 else np.array([0.0])
        return {
            "parca": len(c),
            "ornek_hizi_hz": round(frames / (t1 - t0)),
            "parca_ornek_p50": int(np.median([f for _, f, _ in c])),
            "gelis_araligi_ms_p50_p90_max": [round(float(np.percentile(iv, 50)), 1),
                                            round(float(np.percentile(iv, 90)), 1), round(float(iv.max()), 1)],
            "rms_dbfs": round(20 * np.log10(max(rms, 1e-9)), 1),
        }


def set_chrome_mute(mute: bool) -> int:
    from pycaw.pycaw import AudioUtilities
    n = 0
    for s in AudioUtilities.GetAllSessions():
        try:  # oturum listesinde kapanmis surecler olabilir
            if s.Process is None or s.Process.name().lower() != "chrome.exe":
                continue
            s.SimpleAudioVolume.SetMute(1 if mute else 0, None)
            n += 1
        except Exception:
            continue
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--mute-test", action="store_true")
    args = ap.parse_args()

    pid = chrome_main_pid()
    meter = Meter()
    tap = ProcessAudioCapture(pid, on_data=meter.on_data)
    tap.start()
    print(f"chrome ana PID: {pid}")
    try:
        time.sleep(0.5)
        if not args.mute_test:
            t0 = time.perf_counter()
            time.sleep(args.seconds)
            print("yakalama:", meter.window(t0, time.perf_counter()))
            return
        t0 = time.perf_counter(); time.sleep(2.0); t1 = time.perf_counter()
        n = set_chrome_mute(True)
        time.sleep(0.3); t2 = time.perf_counter(); time.sleep(2.0); t3 = time.perf_counter()
        set_chrome_mute(False)
        time.sleep(0.3); t4 = time.perf_counter(); time.sleep(1.5); t5 = time.perf_counter()
        print(f"chrome ses oturumu: {n}")
        print("A sessize almadan :", meter.window(t0, t1))
        print("B sessizdeyken    :", meter.window(t2, t3))
        print("C geri acilinca   :", meter.window(t4, t5))
    finally:
        if args.mute_test:
            set_chrome_mute(False)  # ne olursa olsun sesi geri ac
        tap.close()


if __name__ == "__main__":
    main()
