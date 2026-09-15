"""A/V senkron test klibi uretir (kendi urettigimiz icerik, TOD degil).

1920x1080 50 FPS: testsrc2 (her kare degisir, WGC tekrar ayiklamasi saati bozmasin) + her 2 sn'de
2 karelik (40 ms) tam beyaz flas. Ses 48 kHz stereo: ayni anlarda 50 ms 1 kHz bip, keskin baslangic.
Klip oynaticida dongude oynatilir (ffplay -loop 0), canli hattan gecirilir; olcum:
tools/av_sync_test.py.

Kullanim:
    .venv/Scripts/python.exe tools/make_av_clip.py --seconds 60
"""
from __future__ import annotations

import argparse
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "av_test_1080p50.mp4"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    flash = "lt(mod(t\\,2)\\,0.04)"
    vf = (f"testsrc2=s=1920x1080:r=50:d={args.seconds},"
          f"drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='{flash}',format=yuv420p")
    beep = "if(lt(mod(t,2),0.05),0.5*sin(2*PI*1000*t),0)"
    af = f"aevalsrc=exprs='{beep}|{beep}':s=48000:d={args.seconds}"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", vf, "-f", "lavfi", "-i", af,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", "50",
           "-c:a", "aac", "-b:a", "192k", "-shortest", args.out]
    subprocess.run(cmd, check=True)
    print(f"{args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
