"""tod_sim zaman eslemesi dogrulamasi: simule kare i, GT karesi round(1,2*i) ile mi eslesiyor?

GT kareleri 1080p'ye kucultulur, her simule kare icin g-2..g+2 arasi PSNR bakilir; en iyi eslesme
tahminle ayni olmali. Ayrica bitrate ve kare sayisi ffprobe ile yazilir.

Kullanim: .venv/Scripts/python.exe tools/check_sim_align.py data/clips/5bgF-5I2P_M.mp4 --start 200
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from make_pairs import RawReader  # noqa: E402
from tod_sim import ffprobe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("--start", type=float, default=200)
    ap.add_argument("--seconds", type=float, default=3)
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()
    sim = os.path.join(ROOT, "data", "sim_tmp", "align_check.mp4")
    os.makedirs(os.path.dirname(sim), exist_ok=True)
    subprocess.run([sys.executable, os.path.join(HERE, "tod_sim.py"), args.gt, "--start", str(args.start),
                    "--seconds", str(args.seconds), "--no-barcode", "--out", sim], check=True, stdout=subprocess.DEVNULL)
    lo = RawReader(sim, 1920, 1080)
    hi = RawReader(args.gt, 1920, 1080, args.start, args.seconds,
                   pre_vf="setpts=N/(60*TB),scale=1920:1080:flags=lanczos:in_color_matrix=bt709:in_range=tv:"
                          "out_color_matrix=bt709:out_range=tv")
    gts = []
    while True:
        f = hi.read()
        if f is None:
            break
        gts.append(f[::2, ::2, 1].astype(np.float32))
    hits, rows = 0, []
    for i in range(args.frames):
        f = lo.read()
        if f is None:
            break
        s = f[::2, ::2, 1].astype(np.float32)
        g = round(1.2 * i)
        cands = [c for c in range(g - 2, g + 3) if 0 <= c < len(gts)]
        ps = {c: 10 * np.log10(255 ** 2 / max(np.mean((s - gts[c]) ** 2), 1e-6)) for c in cands}
        best = max(ps, key=ps.get)
        hits += best == g
        rows.append((i, g, best, round(ps[g], 2), round(sorted(ps.values())[-2], 2)))
    lo.close()
    hi.close()
    pr = ffprobe(sim)
    print(json.dumps({"dogru_eslesme": f"{hits}/{len(rows)}", "ornek (i, g, en_iyi, psnr_g, ikinci)": rows[:12],
                      "kbps": round(int(pr["format"]["bit_rate"]) / 1000), "codec": pr["streams"][0]["codec_name"],
                      "profil": pr["streams"][0].get("profile"), "fps": pr["streams"][0]["r_frame_rate"],
                      "boyut": [pr["streams"][0]["width"], pr["streams"][0]["height"]]}, ensure_ascii=False))
    os.remove(sim)
    os.remove(sim.replace(".mp4", ".json"))


if __name__ == "__main__":
    main()
