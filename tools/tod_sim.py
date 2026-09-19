"""TOD simulatoru (Hat 2.1): temiz 4K60 CC klibi -> TOD benzeri 1080p50 H.264.

TOD olcumu (reports/2026-09-15-faz0-kesif-tod-yakalama.md): 1920x1080 50p, H.264 High, ~4,8 Mbps,
BT.709 limited. Bu arac ayni bozulmayi CC klip uzerinde uretir; canli olcumde (watch --score)
oynaticida oynatilir ve cikis, klibin kendi 4K60 karesiyle (GT) karsilastirilir.

Zaman eslemesi: GT 60 FPS, kaynak 50 FPS. Kaynak karesi i, GT karesi round(1,2*i)'nin
kucultulmus halidir (fps=50:round=near). i % 5 == 0 olan kareler GT ile tam ayni andadir.
Cikis tiki kaynak zamani (i + alpha) karesini gosterir -> GT karesi g = 1,2 * (i + alpha).

Kare numarasi seridi: ust 16 satirda 24 hucre (64 px): 20 bit kare numarasi + 4 bit
(bit sayisi mod 16) denetim. Puanlama bu seridi (4K'da ust 48 satir) disarida birakir.

GT 59,94 FPS ise kareleri sirayla tam 60 FPS kabul edilir (setpts).

Kullanim:
    .venv/Scripts/python.exe tools/tod_sim.py data/clips/63pDvVidZJA.mp4 --start 120 --seconds 10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from upscaler.score import BITS, CELL_W, CHECK_BITS, STRIP_H, read_barcode  # noqa: E402,F401
GT_FPS = 60
SIM_FPS = 50
TARGET_KBPS = 4800


def barcode_filter() -> str:
    """ffmpeg drawbox zinciri: siyah serit + bit basina beyaz hucre (n: cikis kare sirasi)."""
    parts = [f"drawbox=x=0:y=0:w={(BITS + CHECK_BITS) * CELL_W + 2 * CELL_W}:h={STRIP_H}:color=black:t=fill"]
    for b in range(BITS):
        parts.append(f"drawbox=x={(b + 1) * CELL_W}:y=0:w={CELL_W}:h={STRIP_H}:color=white:t=fill:"
                     f"enable='mod(floor(n/{2 ** b})\\,2)'")
    # Denetim: 20 bitin bit sayisi mod 16. ffmpeg ifadesinde popcount yok: toplam ifade.
    pop = "+".join(f"mod(floor(n/{2 ** b})\\,2)" for b in range(BITS))
    for c in range(CHECK_BITS):
        parts.append(f"drawbox=x={(BITS + 1 + c) * CELL_W}:y=0:w={CELL_W}:h={STRIP_H}:color=white:t=fill:"
                     f"enable='mod(floor(mod({pop}\\,16)/{2 ** c})\\,2)'")
    return ",".join(parts)


def clean_vf(pre: float = 1.0, sharp: float = 0.0) -> str:
    """4K60 GT -> sikistirmadan onceki 1080p50 yuv420p (kodlayiciya giren kare).

    F27 onarim verisi hedefi bu zincirin ciktisidir: girdi ile hedef arasindaki tek fark H.264."""
    color = "in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=tv"
    if pre < 0.999:
        # F26: gercek TOD kamera goruntusu 1080p'den yumusak (yapim zinciri). Once kucult, sonra 1080p'ye buyut.
        pw, ph = 2 * round(1920 * pre / 2), 2 * round(1080 * pre / 2)
        scale = f"scale={pw}:{ph}:flags=area:{color},scale=1920:1080:flags=bicubic"
    else:
        scale = f"scale=1920:1080:flags=lanczos:{color}"
    if sharp > 0:
        scale += f",unsharp=5:5:{sharp}:5:5:0"
    return f"setpts=N/({GT_FPS}*TB),fps={SIM_FPS}:round=near,{scale},format=yuv420p"


def ffprobe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,profile,width,height,r_frame_rate,bit_rate,nb_frames,pix_fmt,"
                          "color_space,color_range:format=duration,bit_rate", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", help="4K60 CC klip")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--kbps", type=int, default=TARGET_KBPS)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-barcode", action="store_true")
    ap.add_argument("--pre", type=float, default=1.0, help="F26: ön kucultme orani (1080p'ye gore, 1 = yok)")
    ap.add_argument("--sharp", type=float, default=0.0, help="F26: 1080p'de unsharp miktari (0 = yok)")
    args = ap.parse_args()
    cid = os.path.splitext(os.path.basename(args.gt))[0]
    out = args.out or os.path.join(ROOT, "data", "sim", f"{cid}_s{int(args.start)}_l{int(args.seconds)}_50p.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    src = ffprobe(args.gt)["streams"][0]
    num, den = map(int, src["r_frame_rate"].split("/"))
    if abs(num / den - GT_FPS) > 0.1:
        raise SystemExit(f"GT {num / den:.3f} FPS; bu arac 60 (ya da 59,94) FPS GT bekliyor")
    # 59,94 FPS GT'nin kareleri sirayla tam 60 FPS sayilir (icerik %0,1 yavaslar). GT bankasi da
    # kareleri sirayla sayar; boylece 10 sn'de 0,6 karelik kayma birikmez.
    vf = clean_vf(args.pre, args.sharp)
    if not args.no_barcode:
        vf += "," + barcode_filter()
    k = args.kbps
    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(args.start), "-t", str(args.seconds),
            "-i", args.gt, "-an", "-vf", vf, "-c:v", "libx264", "-profile:v", "high", "-preset", "medium",
            "-b:v", f"{k}k", "-maxrate", f"{int(k * 1.25)}k", "-bufsize", f"{2 * k}k", "-g", "100", "-bf", "3",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
    passlog = out + ".pass"
    subprocess.run(base + ["-pass", "1", "-passlogfile", passlog, "-f", "null", "NUL"], check=True)
    subprocess.run(base + ["-pass", "2", "-passlogfile", passlog, out], check=True)
    for f in os.listdir(os.path.dirname(out)):
        if f.startswith(os.path.basename(passlog)):
            os.remove(os.path.join(os.path.dirname(out), f))
    probe = ffprobe(out)
    st = probe["streams"][0]
    meta = {"gt": os.path.relpath(args.gt, ROOT).replace("\\", "/"), "gt_start_s": args.start, "seconds": args.seconds,
            "gt_fps": GT_FPS, "sim_fps": SIM_FPS, "barcode": not args.no_barcode, "strip_h_1080": STRIP_H,
            "hedef_kbps": k, "pre": args.pre, "sharp": args.sharp, "ffprobe": {"codec": st["codec_name"], "profile": st.get("profile"),
                                         "size": [st["width"], st["height"]], "fps": st["r_frame_rate"],
                                         "kbps": round(int(probe["format"]["bit_rate"]) / 1000),
                                         "frames": st.get("nb_frames"), "color": [st.get("color_space"), st.get("color_range")]}}
    with open(out.replace(".mp4", ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
