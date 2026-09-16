"""Canli olcum (Hat 1.3): simule TOD klibi ffplay'de doner, watch ayni canli hattan gecirir ve puanlar.

ffplay renk donusumu acikca BT.709 (sinirli -> tam aralik) yapilir; GT bankasi da ayni donusumu
kullanir (upscaler/score.py). Ses kapali (puan goruntu icin).

Kullanim:
    .venv/Scripts/python.exe tools/live_score.py data/sim/63pDvVidZJA_s120_l10_50p.json --proc fused-sr --seconds 60
    ... --shift 1        # kasitli 1 GT kare kaydirma: puan belirgin dusmeli
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from av_sync_test import find_window  # noqa: E402

TITLE = "upscaler-score-test"


def launch(clip: str) -> subprocess.Popen:
    env = dict(os.environ, SDL_WINDOWS_DPI_AWARENESS="permonitorv2")
    vf = "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24"
    return subprocess.Popen(["ffplay", "-hide_banner", "-loglevel", "error", "-loop", "0", "-an",
                             "-window_title", TITLE, "-noborder", "-left", "0", "-top", "0",
                             "-x", "1920", "-y", "1080", "-vf", vf, clip], env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("meta", help="tools/tod_sim.py ciktisi .json")
    ap.add_argument("--proc", default="fused-sr")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--run-name", default="")
    ap.add_argument("extra", nargs="*", help="watch'a aynen gecer (-- sonrasinda)")
    args = ap.parse_args()
    meta = os.path.abspath(args.meta)
    clip = meta[:-5] + ".mp4"
    name = args.run_name or f"score_{os.path.basename(meta)[:-5]}_{args.proc}_k{args.shift}"
    player = launch(clip)
    try:
        if find_window(TITLE) is None:
            raise SystemExit("ffplay penceresi acilmadi")
        time.sleep(1.0)
        cmd = [sys.executable, "-m", "upscaler", "watch", "--title", TITLE, "--title-must", "", "--no-audio",
               "--proc", args.proc, "--score", meta, "--score-shift", str(args.shift),
               "--seconds", str(args.seconds), "--run-name", name] + args.extra
        rc = subprocess.call(cmd, cwd=ROOT, stdout=subprocess.DEVNULL)
    finally:
        player.kill()
    with open(os.path.join(ROOT, "runs", name, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    out = {"ad": name, "cikis_kodu": rc, "islemci": args.proc, "cikis_fps_aktif": s.get("cikis_fps_aktif"),
           "gec_tik_orani": s.get("gec_tik_orani"), "islem_ms_p50_p95_p99": s.get("islem_ms_p50_p95_p99"),
           "uctan_uca_ms_p50_p95_p99": s.get("uctan_uca_ms_p50_p95_p99"), "puan": s.get("puan")}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
