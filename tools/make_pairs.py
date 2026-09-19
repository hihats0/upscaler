"""Egitim ciftleri (Hat 2.2 / C1): 4K GT <-> tod_sim bozulmus 1080p, rastgele yamalar.

Her klipten birkac segment secilir; segment tools/tod_sim.py ile (seritsiz) 1080p50 H.264'e
cevrilir, iki akis ayni anda cozulur. Simule kare i, GT karesi round(1,2*i)'nin kucultulmus ve
sikistirilmis halidir (fps filtresi kare secer, ara kare uretmez; tools/check_sim_align.py dogrular).
Her N. kareden K yama alinir: LR PxP, HR 2Px2P, RGB uint8. Yamalar sikistirilmis npz parcalarina
yazilir (data/pairs/<ad>/shard_XXX.npz). Kare dosyasi yazilmaz; simule mp4'ler segment bitince silinir.

Kullanim:
    .venv/Scripts/python.exe tools/make_pairs.py --clips 63pDvVidZJA u5w_du22wSQ --name train1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLIPS = os.path.join(ROOT, "data", "clips")
RGB_VF = "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24"


class RawReader:
    def __init__(self, path: str, w: int, h: int, start: float = 0.0, seconds: float = 0.0, pre_vf: str = "") -> None:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if start:
            cmd += ["-ss", str(start)]
        if seconds:
            cmd += ["-t", str(seconds)]
        vf = (pre_vf + "," if pre_vf else "") + RGB_VF
        cmd += ["-i", path, "-an", "-vf", vf, "-f", "rawvideo", "pipe:1"]
        self.n = w * h * 3
        self.shape = (h, w, 3)
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=self.n,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def read(self) -> np.ndarray | None:
        buf = self.p.stdout.read(self.n)
        if buf is None or len(buf) < self.n:
            return None
        return np.frombuffer(buf, np.uint8).reshape(self.shape)

    def close(self) -> None:
        self.p.stdout.close()
        self.p.kill()
        self.p.wait()


def clip_path(cid: str) -> str:
    import glob
    hits = [p for p in glob.glob(os.path.join(CLIPS, cid + ".*")) if p.endswith((".mp4", ".webm", ".mkv"))]
    if not hits:
        raise SystemExit(f"klip yok: {cid}")
    return hits[0]


def duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def flat_ok(hr: np.ndarray) -> bool:
    """Duz (bilgi tasimayan) yamalari at: gri tonda standart sapma."""
    return float(hr[::4, ::4].astype(np.float32).std()) > 6.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, help="data/clips altindaki klip kimlikleri")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seg-seconds", type=float, default=8.0)
    ap.add_argument("--seg-every", type=float, default=45.0, help="segment baslangiclari arasi (sn)")
    ap.add_argument("--skip-start", type=float, default=20.0, help="klibin basindaki jenerik/logo")
    ap.add_argument("--frame-every", type=int, default=4)
    ap.add_argument("--patches", type=int, default=6, help="kare basina yama")
    ap.add_argument("--lr", type=int, default=128)
    ap.add_argument("--shard", type=int, default=800)
    ap.add_argument("--kbps", type=int, default=4800)
    ap.add_argument("--max-gb", type=float, default=12.0)
    ap.add_argument("--pre-range", default="1:1", help="F26: segment basina rastgele ön kucultme araligi (ör. 0.5:0.85)")
    ap.add_argument("--sharp-range", default="0:0", help="F26: segment basina rastgele unsharp araligi")
    ap.add_argument("--gt-sharp", default="", help="F26: tools/gt_sharpness.py ciktisi; yumusak GT segmentleri atlanir")
    ap.add_argument("--gt-min", type=float, default=0.0, help="F26: 4K spk_50_75 alt siniri")
    ap.add_argument("--target", default="4k", choices=["4k", "1080"],
                    help="F27: 1080 = hedef ayni karenin sikistirilmamis 1080p hali (onarim verisi, LR=HR boyutu)")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "data", "pairs", args.name)
    os.makedirs(out_dir, exist_ok=True)
    sim_dir = os.path.join(ROOT, "data", "sim_tmp")
    os.makedirs(sim_dir, exist_ok=True)
    rng = np.random.default_rng(1234)
    P = args.lr
    Q = P if args.target == "1080" else 2 * P
    sc = 1 if args.target == "1080" else 2
    lr_buf, hr_buf, src_buf = [], [], []
    shard_i = len([f for f in os.listdir(out_dir) if f.startswith("shard_")])
    total_bytes = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    info = {"clips": {}, "args": vars(args)}
    pre_lo, pre_hi = map(float, args.pre_range.split(":"))
    sh_lo, sh_hi = map(float, args.sharp_range.split(":"))
    gt_sharp = {}
    if args.gt_sharp:
        gt_sharp = json.load(open(args.gt_sharp, encoding="utf-8"))["klipler"]

    def flush():
        nonlocal shard_i, total_bytes, lr_buf, hr_buf, src_buf
        if not lr_buf:
            return
        path = os.path.join(out_dir, f"shard_{shard_i:03d}.npz")
        np.savez_compressed(path, lr=np.stack(lr_buf), hr=np.stack(hr_buf), src=np.array(src_buf))
        total_bytes += os.path.getsize(path)
        print(f"  {os.path.basename(path)}: {len(lr_buf)} yama, toplam {total_bytes / 1e9:.2f} GB", flush=True)
        shard_i += 1
        lr_buf, hr_buf, src_buf = [], [], []

    for cid in args.clips:
        gt = clip_path(cid)
        dur = duration(gt)
        starts = np.arange(args.skip_start, dur - args.seg_seconds - 5, args.seg_every)
        n_clip = 0
        n_skip = 0
        t_clip = time.perf_counter()
        for s in starts:
            if total_bytes / 1e9 > args.max_gb:
                break
            seg_sharp = gt_sharp.get(cid, {}).get("segment", {}).get(str(int(s)), {}).get("spk_50_75")
            if args.gt_min and (seg_sharp is None or seg_sharp < args.gt_min):
                n_skip += 1
                continue
            pre = float(rng.uniform(pre_lo, pre_hi))
            sharp = float(rng.uniform(sh_lo, sh_hi))
            sim = os.path.join(sim_dir, f"{cid}_{int(s)}.mp4")
            subprocess.run([sys.executable, os.path.join(HERE, "tod_sim.py"), gt, "--start", str(s), "--seconds",
                            str(args.seg_seconds), "--no-barcode", "--kbps", str(args.kbps), "--out", sim,
                            "--pre", f"{pre:.3f}", "--sharp", f"{sharp:.3f}"],
                           check=True, stdout=subprocess.DEVNULL)
            lo = RawReader(sim, 1920, 1080)
            if args.target == "1080":
                # Ayni filtre zinciri, kodlayicisiz: sim kare i <-> temiz kare i (fps secimi ayni)
                from tod_sim import clean_vf
                hi = RawReader(gt, 1920, 1080, s, args.seg_seconds, pre_vf=clean_vf(pre, sharp))
            else:
                hi = RawReader(gt, 3840, 2160, s, args.seg_seconds, pre_vf="setpts=N/(60*TB)")
            g_next = 0
            hr_frame = None
            i = 0
            while True:
                lr = lo.read()
                if lr is None:
                    break
                g = i if args.target == "1080" else round(1.2 * i)
                while g_next <= g:
                    hr_frame = hi.read()
                    g_next += 1
                    if hr_frame is None:
                        break
                if hr_frame is None:
                    break
                if i % args.frame_every == 0:
                    for _ in range(args.patches):
                        for _try in range(5):
                            y = int(rng.integers(0, 1080 - P)) // 2 * 2
                            x = int(rng.integers(0, 1920 - P)) // 2 * 2
                            h = hr_frame[sc * y:sc * y + Q, sc * x:sc * x + Q]
                            if flat_ok(h):
                                lr_buf.append(lr[y:y + P, x:x + P].copy())
                                hr_buf.append(h.copy())
                                src_buf.append(f"{cid}:{s:.0f}:{i}:p{pre:.2f}:u{sharp:.2f}")
                                n_clip += 1
                                break
                        if len(lr_buf) >= args.shard:
                            flush()
                i += 1
            lo.close()
            hi.close()
            os.remove(sim)
            os.remove(sim.replace(".mp4", ".json"))
        info["clips"][cid] = {"yama": n_clip, "atlanan_segment": n_skip, "sn": round(time.perf_counter() - t_clip)}
        print(f"{cid}: {n_clip} yama, {n_skip} yumusak segment atlandi, {time.perf_counter() - t_clip:.0f} sn", flush=True)
    flush()
    with open(os.path.join(out_dir, f"info_{int(time.time())}.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
