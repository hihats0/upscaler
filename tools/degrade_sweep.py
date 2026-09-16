"""Bozulma taramasi (F26): hangi TOD simulasyonu canli TOD olcumune (tools/sharpness_probe.py) en yakin?

Her CC klip segmenti icin varyant: 4K -> (ön kucultme p) -> 1080p'ye buyutme -> (keskinlestirme) ->
H.264 4,8 Mbps. Kodlanan gecici mp4 olculur ve silinir. Sadece sayi yazilir.

Kullanim:
    .venv/Scripts/python.exe tools/degrade_sweep.py --target runs/tod_net1/ozet.json --name sweep1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402

from make_pairs import clip_path, duration  # noqa: E402
from sharpness_probe import probe_file  # noqa: E402
from upscaler.sharpness import summarize  # noqa: E402

KEYS = ("kayip05", "oran60", "spk_50_75", "spk_75_100")


def variant_vf(p: float, sharp: float, up: str = "bicubic") -> str:
    """p: ön kucultme (1080p'ye gore), sharp: unsharp miktari (0 = yok)."""
    chain = []
    if p < 0.999:
        w, h = 2 * round(1920 * p / 2), 2 * round(1080 * p / 2)
        chain.append(f"scale={w}:{h}:flags=area")
    chain.append(f"scale=1920:1080:flags={'lanczos' if p >= 0.999 else up}")
    if sharp > 0:
        chain.append(f"unsharp=5:5:{sharp}:5:5:0")
    return ",".join(chain)


def encode(gt: str, start: float, seconds: float, vf: str, kbps: int, out: str) -> None:
    full = f"fps=50:round=near,{vf},format=yuv420p"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-t", str(seconds),
                    "-i", gt, "-an", "-vf", full, "-c:v", "libx264", "-profile:v", "high", "-preset", "medium",
                    "-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.25)}k", "-bufsize", f"{2 * kbps}k",
                    "-g", "100", "-bf", "3", out], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--clips", nargs="+", default=["63pDvVidZJA", "u5w_du22wSQ", "5bgF-5I2P_M", "QdLKQ3SnHmc"])
    ap.add_argument("--segs", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--kbps", type=int, default=4800)
    ap.add_argument("--variants", default="1:0,0.85:0,0.75:0,0.67:0,0.5:0,0.75:0.6,0.67:0.6,0.67:1.2,0.5:1.2")
    args = ap.parse_args()
    target = json.load(open(os.path.join(ROOT, args.target), encoding="utf-8"))["ozet"]
    variants = [tuple(map(float, v.split(":"))) for v in args.variants.split(",")]
    tmp = os.path.join(ROOT, "data", "sim_tmp")
    os.makedirs(tmp, exist_ok=True)
    out_dir = os.path.join(ROOT, "runs", args.name)
    os.makedirs(out_dir, exist_ok=True)
    rows_by_v: dict[str, list[dict]] = {}
    per_seg = []
    t0 = time.perf_counter()
    for cid in args.clips:
        gt = clip_path(cid)
        dur = duration(gt)
        starts = np.linspace(30, dur - args.seconds - 10, args.segs)
        for s in starts:
            for p, sh in variants:
                key = f"p{p:g}_s{sh:g}"
                mp4 = os.path.join(tmp, f"sweep_{cid}_{int(s)}_{key}.mp4")
                encode(gt, float(s), args.seconds, variant_vf(p, sh), args.kbps, mp4)
                rows = probe_file(mp4, 10, 0.0, 0.0, "")
                os.remove(mp4)
                rows_by_v.setdefault(key, []).extend(rows)
                sm = summarize(rows)
                per_seg.append({"klip": cid, "bas": int(s), "varyant": key, **{k: sm.get(k) for k in KEYS}})
                print(f"{time.perf_counter() - t0:6.0f}s {cid} {int(s)} {key}: " +
                      " ".join(f"{k}={sm.get(k)}" for k in KEYS), flush=True)
    table = {}
    for key, rows in rows_by_v.items():
        sm = summarize(rows)
        # hedefe uzaklik: her olcu icin log oran farki
        dist = float(np.sqrt(np.mean([np.log(sm[k] / target[k]) ** 2 for k in KEYS])))
        table[key] = {**{k: sm[k] for k in KEYS}, "uzaklik": round(dist, 4)}
    best = sorted(table.items(), key=lambda kv: kv[1]["uzaklik"])
    res = {"hedef": {k: target[k] for k in KEYS}, "varyantlar": dict(best), "segmentler": per_seg}
    with open(os.path.join(out_dir, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("hedef", res["hedef"])
    for k, v in best:
        print(k, v)


if __name__ == "__main__":
    main()
