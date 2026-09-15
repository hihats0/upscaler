"""A/V senkron olcumu (Hat 1.4 paket 3).

Kendi urettigimiz flas + bip klibi (tools/make_av_clip.py) ffplay'de dongude oynatilir, pencere ve
sesi `python -m upscaler watch --av-measure` ile ayni canli hattan gecer. Watch girdide (WGC karesi,
ProcTap sesi) ve cikista (sunulan kare, DAC zamanli ses blogu) flas/bip baslangiclarini toplar.

Kullanim:
    .venv/Scripts/python.exe tools/av_sync_test.py --seconds 180 --run-name av_3dk
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
sys.path.insert(0, ROOT)
TITLE = "upscaler-av-test"


def find_window(title: str, timeout: float = 15.0) -> int | None:
    import win32gui
    t0 = time.time()
    while time.time() - t0 < timeout:
        hits = []
        win32gui.EnumWindows(lambda h, _: hits.append(h) if win32gui.IsWindowVisible(h)
                             and win32gui.GetWindowText(h) == title else None, None)
        if hits:
            return hits[0]
        time.sleep(0.2)
    return None


def launch_player(clip: str, w: int = 1920, h: int = 1080) -> subprocess.Popen:
    env = dict(os.environ, SDL_WINDOWS_DPI_AWARENESS="permonitorv2")
    return subprocess.Popen(["ffplay", "-hide_banner", "-loglevel", "error", "-loop", "0",
                             "-window_title", TITLE, "-noborder", "-left", "0", "-top", "0",
                             "-x", str(w), "-y", str(h), clip], env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--run-name", default=time.strftime("av_%Y%m%d_%H%M%S"))
    ap.add_argument("--clip", default=os.path.join(ROOT, "data", "av_test_1080p50.mp4"))
    ap.add_argument("extra", nargs="*", help="watch'a aynen gecer (-- sonrasinda)")
    args = ap.parse_args()
    if not os.path.exists(args.clip):
        subprocess.run([sys.executable, os.path.join(HERE, "make_av_clip.py")], check=True)
    player = launch_player(args.clip)
    try:
        if find_window(TITLE) is None:
            raise SystemExit("ffplay penceresi acilmadi")
        time.sleep(1.0)
        cmd = [sys.executable, "-m", "upscaler", "watch", "--title", TITLE, "--title-must", "",
               "--av-measure", "--seconds", str(args.seconds), "--run-name", args.run_name] + args.extra
        rc = subprocess.call(cmd, cwd=ROOT)
    finally:
        player.kill()
    path = os.path.join(ROOT, "runs", args.run_name, "summary.json")
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    print(f"watch cikis kodu {rc}")
    print(json.dumps({"av": s.get("av"), "ses": s.get("ses"), "cikis_fps_aktif": s.get("cikis_fps_aktif"),
                      "gec_tik": s.get("gec_tik"), "uctan_uca_ms_p50_p95_p99": s.get("uctan_uca_ms_p50_p95_p99")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
