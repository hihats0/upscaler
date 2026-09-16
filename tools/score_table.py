"""Canli puan kosularindan kiyas tablosu (Hat 1.3 / B4).

Her kosunun puan_kareler.json satirlari (t, g, sinif, alpha, psnr, ssim) okunur. Adil kiyas icin
her sinifta TUM kosularin ortak GT kareleri (g) uzerinden ortalama alinir (ayni kare birden cok
puanlandiysa ortalamasi). Markdown tablo yazar.

Kullanim: .venv/Scripts/python.exe tools/score_table.py kiyas_s200_baseline kiyas_s200_sr ...
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name: str):
    d = os.path.join(ROOT, "runs", name)
    with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    with open(os.path.join(d, "puan_kareler.json"), encoding="utf-8") as f:
        rows = json.load(f)
    per = {"gercek": defaultdict(list), "ara": defaultdict(list)}
    for t, g, cls, alpha, p, ss in rows:
        per[cls][(s["puan"]["gt"], s["puan"]["gt_start_s"], g)].append((p, ss))
    return s, {c: {k: np.mean(v, axis=0) for k, v in m.items()} for c, m in per.items()}


def main() -> None:
    names = sys.argv[1:]
    data = [load(n) for n in names]
    common = {}
    for cls in ("gercek", "ara"):
        keys = [set(d[1][cls]) for d in data if d[1][cls]]
        common[cls] = set.intersection(*keys) if len(keys) == len(data) else set()
    print("| Kosu | Islemci | SR | Cikis FPS | Gec tik | Islem p95 ms | Gercek PSNR / SSIM | Ara PSNR / SSIM |")
    print("|---|---|---|---|---|---|---|---|")
    for n, (s, per) in zip(names, data):
        cells = []
        for cls in ("gercek", "ara"):
            ks = common[cls]
            if ks:
                arr = np.array([per[cls][k] for k in sorted(ks)])
                cells.append(f"{arr[:, 0].mean():.2f} / {arr[:, 1].mean():.4f} (n={len(ks)})")
            else:
                cells.append("-")
        cfg = {}
        ev = os.path.join(ROOT, "runs", n, "events.jsonl")
        with open(ev, encoding="utf-8") as f:
            cfg = json.loads(f.readline())["ayarlar"]
        print(f"| {n} | {cfg['proc']} | {cfg['sr'] if cfg['proc'] in ('sr', 'fused-sr', 'rife-flow-trt-sr') else 'bicubic'} | {s['cikis_fps_aktif']} | %{100 * s['gec_tik_orani']:.3f} | "
              f"{s['islem_ms_p50_p95_p99'][1]} | {cells[0]} | {cells[1]} |")


if __name__ == "__main__":
    main()
