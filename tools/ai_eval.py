"""AI yeniden cizim adaylarinin kalite olcumu (goal 2026-09-20, Asama A/C).

Girdi: simule TOD klibi (tod_sim, 4,8 Mbps, on kucultme 0,7 + unsharp 0,4 = canli TOD'a yakin).
Hedef: ayni anin 4K CC karesinden KESKIN 1080p (lanczos, yumusatma yok; tod_sim.clean_vf(1, 0)).
Sadece dogrulama klipleri (5bgF-5I2P_M, O3gD6n0zoik; egitimde yok). TOD karesi kullanilmaz.

Olculer (kadrajin ortasi, upscaler/sharpness.CROP):
- psnr: Y kanali PSNR (hedefe).
- lpips: algisal uzaklik (AlexNet, dusuk = iyi).
- spk_50_75 / spk_75_100: sharpness.measure (medyan). Hedefin spk'si de yazilir.
- titreme: statik bolgede (hedefin ardisik kare farki kucuk) ardisik kare farki, cikis / hedef.

    .venv/Scripts/python.exe tools/ai_eval.py --cands ham bicubic-540 esr-gen-270 --name tarama1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from make_pairs import RawReader, clip_path  # noqa: E402
from tod_sim import clean_vf  # noqa: E402
from upscaler.sharpness import CROP, luma, measure  # noqa: E402

CLIPS = [  # (simule klip, GT klip, GT baslangic sn)
    ("data/sim/5bgF-5I2P_M_s20_l10_50p_tod.mp4", "5bgF-5I2P_M", 20.0),
    ("data/sim/5bgF-5I2P_M_s200_l10_50p_tod.mp4", "5bgF-5I2P_M", 200.0),
    ("data/sim/O3gD6n0zoik_s20_l10_50p_tod.mp4", "O3gD6n0zoik", 20.0),
]


def crop(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    return a[int(h * CROP[0]):int(h * CROP[1]), int(w * CROP[2]):int(w * CROP[3])]


class Runner:
    """Aday adi -> fonksiyon(list[RGB uint8 HxWx3] kareler t-k..t+k) -> RGB uint8 cikis (orta kare)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.frames = 1
        self.blend = None
        if "+blend" in name:  # ör. ai_v2+blend0.6: cikarimda durgun bolge harmani (StaticBlend)
            name, k = name.split("+blend")
            from upscaler.models.ai import StaticBlend
            self.blend = StaticBlend(float(k or 0.6))
        if name == "ham":
            self.net = None
            return
        from upscaler.models.ai import build
        self.net = build(name)
        self.frames = getattr(self.net, "frames", 1)
        self.fp32 = name.startswith("swinir")
        if not self.fp32:
            self.net = self.net.half()

    @torch.inference_mode()
    def __call__(self, win: list[np.ndarray]) -> np.ndarray:
        mid = win[len(win) // 2]
        if self.net is None:
            return mid
        k = self.frames // 2
        c = len(win) // 2
        sel = win[c - k:c + k + 1]
        x = torch.cat([torch.from_numpy(f).cuda().permute(2, 0, 1)[None] for f in sel], 1)
        x = x.float() / 255
        if not self.fp32:
            x = x.half()
        y = self.net(x).float().clamp(0, 1)
        if self.blend is not None:
            y = self.blend(x[:, k * 3:k * 3 + 3].float(), y)
        return (y[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()
    import lpips
    lp = None if args.no_lpips else lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    runners = [Runner(c) for c in args.cands]
    R = 2  # pencere yaricapi (t-2..t+2); aday kac kare isterse ortadan alir
    acc = {c: {"psnr": [], "lpips": [], "spk": [], "fl_out": 0.0, "fl_gt": 0.0, "ms": []} for c in args.cands}
    tgt_spk = []
    t0 = time.perf_counter()
    for sim, cid, start in CLIPS:
        lo = RawReader(os.path.join(ROOT, sim), 1920, 1080)
        hi = RawReader(clip_path(cid), 1920, 1080, start, args.seconds, pre_vf=clean_vf(1.0, 0.0))
        win_lo, win_hi = [], []
        prev = {}
        i = 0
        while True:
            a, b = lo.read(), hi.read()
            if a is None or b is None:
                break
            win_lo = (win_lo + [a])[-(2 * R + 1):]
            win_hi = (win_hi + [b])[-(2 * R + 1):]
            i += 1
            t = i - 1 - R  # pencerenin orta karesinin numarasi
            if len(win_lo) < 2 * R + 1:
                continue
            if t % args.every not in (0, 1):
                for r in runners:  # harmanli aday onceki cikisa bagli: her kareyi islemeli
                    if r.blend is not None:
                        r(win_lo)
                continue
            tgt = win_hi[R]
            ty = crop(luma(tgt, bgr=False))
            if t % args.every == 0:
                tgt_spk.append(measure(luma(tgt, bgr=False)))
            for r in runners:
                torch.cuda.synchronize()
                ts = time.perf_counter()
                out = r(win_lo)
                acc[r.name]["ms"].append((time.perf_counter() - ts) * 1000)
                oy = crop(luma(out, bgr=False))
                if t % args.every == 0:
                    mse = float(((oy - ty) ** 2).mean())
                    acc[r.name]["psnr"].append(10 * np.log10(255 ** 2 / max(mse, 1e-6)))
                    acc[r.name]["spk"].append(measure(luma(out, bgr=False)))
                    if lp is not None:
                        o_t = torch.from_numpy(crop(out)).cuda().permute(2, 0, 1)[None].float() / 127.5 - 1
                        g_t = torch.from_numpy(crop(tgt)).cuda().permute(2, 0, 1)[None].float() / 127.5 - 1
                        with torch.no_grad():
                            acc[r.name]["lpips"].append(float(lp(o_t, g_t)))
                    prev[r.name] = (oy, ty)
                elif r.name in prev:  # t = k*every + 1: ardisik kare titremesi
                    oy0, ty0 = prev.pop(r.name)
                    dg = cv2.GaussianBlur(np.abs(ty - ty0), (0, 0), 3)
                    mask = dg < 1.0
                    if mask.mean() > 0.05:
                        acc[r.name]["fl_out"] += float(np.abs(oy - oy0)[mask].sum())
                        acc[r.name]["fl_gt"] += float(np.abs(ty - ty0)[mask].sum())
        lo.close()
        hi.close()
        print(f"{cid}@{start:.0f}: {time.perf_counter() - t0:.0f} sn", flush=True)
    res = {"hedef": {"spk_50_75": float(np.median([m["spk_50_75"] for m in tgt_spk])),
                     "spk_75_100": float(np.median([m["spk_75_100"] for m in tgt_spk])), "n": len(tgt_spk)},
           "adaylar": {}}
    for c, a in acc.items():
        res["adaylar"][c] = {
            "psnr": round(float(np.mean(a["psnr"])), 3),
            "lpips": round(float(np.mean(a["lpips"])), 4) if a["lpips"] else None,
            "spk_50_75": round(float(np.median([m["spk_50_75"] for m in a["spk"]])), 4),
            "spk_75_100": round(float(np.median([m["spk_75_100"] for m in a["spk"]])), 4),
            "titreme": round(a["fl_out"] / max(a["fl_gt"], 1e-6), 3),
            "pytorch_ms_p50": round(float(np.median(a["ms"])), 1),
        }
    os.makedirs(os.path.join(ROOT, "runs", "ai_eval"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", "ai_eval", f"{args.name}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    h = res["hedef"]
    print(f"hedef spk_50_75 {h['spk_50_75']:.4f}  spk_75_100 {h['spk_75_100']:.4f}  (n={h['n']})")
    print(f"{'aday':<22}{'PSNR':>8}{'LPIPS':>8}{'spk50':>8}{'spk75':>8}{'titr':>7}{'ms':>8}")
    for c, r in res["adaylar"].items():
        print(f"{c:<22}{r['psnr']:>8.2f}{(r['lpips'] or 0):>8.4f}{r['spk_50_75']:>8.4f}{r['spk_75_100']:>8.4f}"
              f"{r['titreme']:>7.3f}{r['pytorch_ms_p50']:>8.1f}")


if __name__ == "__main__":
    main()
