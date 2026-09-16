"""CC 4K kliplerin bolum bolum gercek keskinligi (F26). Sadece sayi yazar.

Her klipte make_pairs ile ayni segment baslangiclari (skip_start + k * seg_every), segment basina
birkac 4K kare olculur (upscaler.sharpness, 4K'da). Yumusak segmentler egitimde hedef olmamali.

Kullanim:
    .venv/Scripts/python.exe tools/gt_sharpness.py --out runs/gt_sharp.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402

from make_pairs import CLIPS, clip_path, duration  # noqa: E402
from upscaler.sharpness import luma, measure  # noqa: E402


def seg_score(gt: str, start: float, seconds: float, frames: int) -> dict:
    w, h = 3840, 2160
    n = w * h * 3
    step = seconds / frames
    vals = []
    for k in range(frames):
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start + k * step:.2f}", "-i", gt,
               "-frames:v", "1", "-an", "-vf", "format=rgb24", "-f", "rawvideo", "pipe:1"]
        buf = subprocess.run(cmd, capture_output=True, check=True).stdout
        if len(buf) < n:
            continue
        img = np.frombuffer(buf[:n], np.uint8).reshape(h, w, 3)
        vals.append(measure(luma(img, bgr=False)))
    if not vals:
        return {}
    return {k: round(float(np.median([v[k] for v in vals])), 4) for k in ("std", "kayip05", "spk_50_75", "spk_75_100")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg-seconds", type=float, default=8.0)
    ap.add_argument("--seg-every", type=float, default=45.0)
    ap.add_argument("--skip-start", type=float, default=20.0)
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()
    ids = sorted({os.path.splitext(f)[0] for f in os.listdir(CLIPS) if f.endswith((".mp4", ".webm"))})
    res = {"not": "4K karede spk_50_75 (Nyquist'in 0,5-0,75 bandi / 0,25-0,5 bandi)", "klipler": {}}
    for cid in ids:
        gt = clip_path(cid)
        dur = duration(gt)
        segs = {}
        for s in np.arange(args.skip_start, dur - args.seg_seconds - 5, args.seg_every):
            segs[str(int(s))] = seg_score(gt, float(s), args.seg_seconds, args.frames)
        sp = [v["spk_50_75"] for v in segs.values() if v]
        res["klipler"][cid] = {"segment": segs, "spk_medyan": round(float(np.median(sp)), 4) if sp else None}
        print(cid, len(segs), "segment, spk medyan", res["klipler"][cid]["spk_medyan"],
              "min", min(sp) if sp else None, "max", max(sp) if sp else None, flush=True)
        with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
