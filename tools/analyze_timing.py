"""Canli koşu zaman kaydini inceler (upscaler.live --dump-timing ciktilari).

Girdi: <ad>.csv (benzersiz kareler: raw_s, barcode, index, t, repeat) ve
       <ad>_ticks.csv (cikis tikleri). Sadece sayilar, goruntu yok.

Kullanim:
    .venv/Scripts/python.exe tools/analyze_timing.py runs/tod_x.csv
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter

import numpy as np


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def pct(xs, *qs):
    return [round(float(np.percentile(xs, q)), 3) for q in qs] if len(xs) else []


def main() -> None:
    path = sys.argv[1]
    frames = load(path)
    stem, ext = os.path.splitext(path)
    ticks = load(f"{stem}_ticks{ext}") if os.path.exists(f"{stem}_ticks{ext}") else []

    raw = np.array([r["raw_s"] for r in frames])
    print(f"== kareler: {len(frames)}")
    if len(raw) > 2:
        span = raw[-1] - raw[0]
        d = np.diff(raw) * 1000
        print(f"benzersiz kare hizi {(len(raw) - 1) / span:.2f} fps, sure {span:.1f} sn")
        print(f"kare arasi ms p1/p50/p99/max: {pct(d, 1, 50, 99, 100)}")
        per = float(np.median(d))
        bins = Counter(int(round(x / 20.0)) for x in d)  # 20 ms (50 FPS) biriminde
        print("kare arasi (20 ms birimi) dagilimi:", dict(sorted(bins.items())))
        # Uzun bosluklar: yayin takilmasi mi?
        gaps = [(round(raw[i] - raw[0], 2), round(x, 1)) for i, x in enumerate(d) if x > 60]
        print(f"60 ms ustu bosluk: {len(gaps)} (ilk 10: {gaps[:10]}) | medyan aralik {per:.2f} ms")
        idx = np.array([r["index"] for r in frames])
        di = np.diff(idx)
        print("sira farki dagilimi:", dict(sorted(Counter(int(x) for x in di).items())))
        print(f"tekrar isaretli: {int(sum(r['repeat'] for r in frames))}")
        t = np.array([r["t"] for r in frames])
        lag = (raw - t) * 1000
        print(f"raw - ideal t (ms) p1/p50/p99: {pct(lag, 1, 50, 99)}")
        if len(raw) > 200:
            w = len(raw) // 5
            for k in range(5):
                seg = raw[k * w:(k + 1) * w]
                print(f"  bolum {k}: {(len(seg) - 1) / (seg[-1] - seg[0]):.2f} fps")

    if not ticks:
        return
    print(f"\n== tikler: {len(ticks)}")
    alpha = np.array([r["alpha"] for r in ticks])
    real = (alpha <= 1e-3) | (alpha >= 1 - 1e-3)
    same = np.array([r["a_index"] == r["b_index"] for r in ticks])
    hold = np.array([r["hold"] for r in ticks], dtype=bool)
    print(f"gercek kare tik orani {real.mean():.3f} | a==b (ayni kare) {same.mean():.3f} | tutma {hold.mean():.3f}")
    print("alpha*6 dagilimi:", dict(sorted(Counter(round(a * 6, 2) for a in alpha).items())[:14]))
    span = np.array([r["b_index"] - r["a_index"] for r in ticks])
    print("b-a sira farki:", dict(sorted(Counter(int(x) for x in span).items())))
    # a_t kaynak izgarasina oturuyor mu? s - a_t, 1/300 sn biriminde
    s = np.array([r["s"] for r in ticks])
    at = np.array([r["a_t"] for r in ticks])
    ph = (s - at) * 300
    print(f"(s - a_t)*300 ondalik kismi p50/p99: {pct(np.abs(ph - np.round(ph)), 50, 99)}")
    pm = np.array([r["proc_ms"] for r in ticks])
    sl = np.array([r["start_late_ms"] for r in ticks])
    print(f"islem ms p50/p95/p99/max: {pct(pm, 50, 95, 99, 100)} | gercek kare p50 {pct(pm[real], 50)} "
          f"ara kare p50/p95 {pct(pm[~real], 50, 95)}")
    print(f"tik baslama gecikmesi ms p50/p99/max: {pct(sl, 50, 99, 100)} | gec tik toplam {int(sum(r['late_before'] for r in ticks))}")
    T = np.array([r["T_s"] for r in ticks])
    if len(T) > 300:
        w = len(T) // 5
        for k in range(5):
            sel = slice(k * w, (k + 1) * w)
            print(f"  bolum {k}: gercek {real[sel].mean():.2f} ayni {same[sel].mean():.2f} tutma {hold[sel].mean():.2f} "
                  f"islem p95 {pct(pm[sel], 95)}")


if __name__ == "__main__":
    main()
