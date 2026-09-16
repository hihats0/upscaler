"""Etkili cozunurluk olcumu (F26): canli pencere ya da video dosyasi.

Pencere modu: WGC ile yakalar, saniyede --rate kare olcer. Kareler bellekte kalir, diske
SADECE sayi yazilir (runs/<ad>/sharpness.csv + ozet.json). TOD kurali: kayit yok.
Dosya modu: kendi CC/simulasyon kliplerimiz (ffmpeg ile cozulur, 1080p'ye sigdirilir).

Kullanim:
    .venv/Scripts/python.exe tools/sharpness_probe.py --window "TOD - Google Chrome" --seconds 120 --name tod_net1
    .venv/Scripts/python.exe tools/sharpness_probe.py --file data/sim/x.mp4 --every 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from upscaler.sharpness import luma, measure, summarize  # noqa: E402


class SrBank:
    """Canli kareyi SR modellerinden ve bicubic'ten gecirir, 4K ciktinin olcusunu dondurur (sadece sayi)."""

    def __init__(self, names: list[str]) -> None:
        import torch

        from upscaler.models.sr import load_sr
        self.torch = torch
        self.nets = {n: load_sr(n).half() for n in names}

    def __call__(self, bgra: np.ndarray) -> dict[str, float]:
        torch = self.torch
        x = torch.from_numpy(bgra[..., 2::-1].copy()).cuda().permute(2, 0, 1)[None].half() / 255.0
        outs = {"bicubic": torch.nn.functional.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)}
        with torch.inference_mode():
            for n, net in self.nets.items():
                outs[n] = net(x)
        res = {}
        for n, y in outs.items():
            img = (y.clamp(0, 1)[0].permute(1, 2, 0).float() * 255).cpu().numpy()
            m = measure(luma(img, bgr=False))
            for k in ("kayip05", "spk_50_75", "spk_75_100"):
                res[f"{n}|{k}"] = m[k]
        return res


def probe_window(title: str, seconds: float, rate: float, front: bool,
                 bank: "SrBank | None" = None) -> tuple[list[dict], dict]:
    from windows_capture import WindowsCapture

    from upscaler.winutil import bring_front, dpi_aware, find_window

    dpi_aware()
    hwnd = find_window(title)
    if not hwnd:
        raise SystemExit(f"pencere yok: {title}")
    if front:
        bring_front(hwnd)
        time.sleep(0.5)
    q: queue.Queue = queue.Queue(maxsize=2)
    info = {"gelen": 0, "boyutlar": {}, "atlanan_boyut": 0}
    last = [0.0]
    prev_small = [None]
    done = threading.Event()
    cap = WindowsCapture(cursor_capture=False, draw_border=False, window_hwnd=hwnd, minimum_update_interval=1)

    @cap.event
    def on_frame_arrived(frame, control):
        if done.is_set():
            control.stop()
            return
        info["gelen"] += 1
        key = f"{frame.width}x{frame.height}"
        info["boyutlar"][key] = info["boyutlar"].get(key, 0) + 1
        now = time.perf_counter()
        if now - last[0] < 1.0 / rate:
            return
        if frame.width != 1920 or not 1076 <= frame.height <= 1080:
            info["atlanan_boyut"] += 1   # tam ekran degil: video kucultulmus olur, olcum gecersiz
            return
        buf = frame.frame_buffer
        small = buf[::16, ::16, 1]
        motion = float(np.abs(small.astype(np.int16) - prev_small[0]).mean()) if prev_small[0] is not None else 0.0
        prev_small[0] = small.astype(np.int16)
        last[0] = now
        try:
            q.put_nowait((now, buf.copy() if bank else luma(buf), motion))
        except queue.Full:
            pass

    @cap.event
    def on_closed():
        pass

    ctl = cap.start_free_threaded()
    rows: list[dict] = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        try:
            t, y, motion = q.get(timeout=0.5)
        except queue.Empty:
            continue
        r = {"t": round(t - t0, 2), "hareket": motion}
        if bank:
            r.update(measure(luma(y)))
            r.update(bank(y))
        else:
            r.update(measure(y))
        rows.append(r)
    done.set()
    try:
        ctl.stop()
    except Exception:
        pass
    return rows, info


def probe_file(path: str, every: int, start: float, seconds: float, vf: str) -> list[dict]:
    w, h = 1920, 1080
    chain = (vf + "," if vf else "") + f"scale={w}:{h}:flags=lanczos,format=rgb24"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-i", path, "-an", "-vf", chain, "-f", "rawvideo", "pipe:1"]
    n = w * h * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=n)
    rows, i, prev = [], 0, None
    while True:
        buf = p.stdout.read(n)
        if not buf or len(buf) < n:
            break
        if i % every == 0:
            img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            small = img[::16, ::16, 1].astype(np.int16)
            r = {"t": i, "hareket": float(np.abs(small - prev).mean()) if prev is not None else 0.0}
            prev = small
            r.update(measure(luma(img, bgr=False)))
            rows.append(r)
        i += 1
    p.wait()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="")
    ap.add_argument("--file", default="")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--rate", type=float, default=2.0, help="pencere: saniyede olculen kare")
    ap.add_argument("--every", type=int, default=10, help="dosya: her N karede bir")
    ap.add_argument("--vf", default="", help="dosya: olcumden once ffmpeg filtresi (deneme)")
    ap.add_argument("--no-front", action="store_true")
    ap.add_argument("--name", default="")
    ap.add_argument("--sr", nargs="*", default=[], help="pencere: bu SR modellerinin 4K ciktisini da olc")
    args = ap.parse_args()
    if args.window:
        bank = SrBank(args.sr) if args.sr else None
        rows, info = probe_window(args.window, args.seconds, args.rate, not args.no_front, bank)
    elif args.file:
        rows, info = probe_file(args.file, args.every, args.start, args.seconds, args.vf), {"dosya": args.file, "vf": args.vf}
    else:
        raise SystemExit("--window ya da --file")
    ozet = summarize(rows)
    moving = summarize([r for r in rows if r["hareket"] >= 4.0])
    res = {"ozet": ozet, "hareketli": moving, "bilgi": info}
    if args.name:
        d = os.path.join(ROOT, "runs", args.name)
        os.makedirs(d, exist_ok=True)
        if rows:
            with open(os.path.join(d, "sharpness.csv"), "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
        with open(os.path.join(d, "ozet.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
