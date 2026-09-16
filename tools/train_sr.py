"""RT4KSR x2 ince ayari (Hat 2.2 / C2): futbol + TOD bozulmasi, laptopta.

Mimari ayni (egitim bicimi rep=False; bitince rep_state_dict ile tek 3x3 -> ayni hiz).
Veri: tools/make_pairs.py parcalari (LR 1080p-olcek yama, HR 4K-olcek yama, RGB uint8).
Guvenlik: her --ckpt-min dakikada last.pt (kaldigi yerden devam), GPU >= --hot C'de
<= --cool C'ye kadar duraklama, fis cekilirse duraklama. Kayit runs/train_<ad>/log.csv.
En iyi dogrulama PSNR'i weights/rt4ksr/<ad>_best.pth (load_sr ile ayni bicim).

Kullanim:
    .venv/Scripts/python.exe tools/train_sr.py --name ft1 --train train1 --val val1 --iters 30000
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from third_party.rt4ksr.arch import RT4KSR, clean_checkpoint, rep_state_dict  # noqa: E402
from upscaler.watch import GpuMonitor  # noqa: E402

PAIRS = os.path.join(ROOT, "data", "pairs")
Y_COEF = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)


def shards(name: str) -> list[str]:
    return sorted(glob.glob(os.path.join(PAIRS, name, "shard_*.npz")))


def load_shard(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["lr"], z["hr"]


class ShardStream:
    """Rastgele parca gruplarindan karistirilmis yigin akisi (RAM'de en fazla `group` parca)."""

    def __init__(self, files: list[str], batch: int, group: int, seed: int) -> None:
        self.files, self.batch, self.group = files, batch, group
        self.rng = np.random.default_rng(seed)
        self._lr = self._hr = None
        self._idx: np.ndarray = np.zeros(0, int)
        self._pos = 0

    def _refill(self) -> None:
        pick = self.rng.choice(len(self.files), size=min(self.group, len(self.files)), replace=False)
        lrs, hrs = zip(*(load_shard(self.files[i]) for i in pick))
        self._lr, self._hr = np.concatenate(lrs), np.concatenate(hrs)
        self._idx = self.rng.permutation(len(self._lr))
        self._pos = 0

    def next(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self._pos + self.batch > len(self._idx):
            self._refill()
        ii = np.sort(self._idx[self._pos:self._pos + self.batch])
        self._pos += self.batch
        lr = torch.from_numpy(self._lr[ii]).to(device, non_blocking=True)
        hr = torch.from_numpy(self._hr[ii]).to(device, non_blocking=True)
        lr = lr.permute(0, 3, 1, 2).float() / 255
        hr = hr.permute(0, 3, 1, 2).float() / 255
        # Ayni rastgele cevirme/donme: yigin genelinde (ucuz)
        k = int(self.rng.integers(4))
        if k:
            lr, hr = torch.rot90(lr, k, (2, 3)), torch.rot90(hr, k, (2, 3))
        if self.rng.random() < 0.5:
            lr, hr = lr.flip(3), hr.flip(3)
        return lr.contiguous(), hr.contiguous()


def psnr_y(sr: torch.Tensor, hr: torch.Tensor, border: int = 4) -> torch.Tensor:
    c = Y_COEF.to(sr.device)
    ys = (sr.clamp(0, 1) * c).sum(1)[:, border:-border, border:-border]
    yh = (hr * c).sum(1)[:, border:-border, border:-border]
    mse = ((ys - yh) ** 2).mean((1, 2)).clamp_min(1e-10)
    return 10 * torch.log10(1 / mse)


@torch.no_grad()
def evaluate(net, val: list[tuple[np.ndarray, np.ndarray]], device: str) -> dict:
    net.eval()
    ps, pb = [], []
    for lr_np, hr_np in val:
        for s in range(0, len(lr_np), 64):
            lr = torch.from_numpy(lr_np[s:s + 64]).to(device).permute(0, 3, 1, 2).float() / 255
            hr = torch.from_numpy(hr_np[s:s + 64]).to(device).permute(0, 3, 1, 2).float() / 255
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sr = net(lr).float()
            bic = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
            ps.append(psnr_y(sr, hr))
            pb.append(psnr_y(bic, hr))
    net.train()
    return {"psnr": float(torch.cat(ps).mean()), "bicubic": float(torch.cat(pb).mean())}


def save_release(net, path: str) -> None:
    """Orijinal checkpoint bicimi ({'state_dict': egitim bicimi}); load_sr bunu okur."""
    sd = {k: v.detach().float().cpu() for k, v in net.state_dict().items()}
    torch.save({"state_dict": sd}, path)
    # Birlestirme dogrulugu: rep bicimi yuklenebilir olmali
    RT4KSR(upscale=2, rep=True).load_state_dict(rep_state_dict(sd), strict=True)


def wait_safe(gpu: GpuMonitor, hot: float, cool: float, log) -> float:
    """Sicaklik ya da fis sorunu varsa duraklar; bekleme suresini dondurur."""
    import psutil
    waited = 0.0
    while True:
        t = gpu.last.get("sicaklik_c")
        bat = psutil.sensors_battery()
        plugged = bat is None or bat.power_plugged
        if plugged and (t is None or t < hot):
            return waited
        reason = "fis cekili" if not plugged else f"GPU {t:.0f} C"
        log(f"duraklama: {reason}")
        while True:
            time.sleep(10)
            waited += 10
            t = gpu.last.get("sicaklik_c")
            bat = psutil.sensors_battery()
            plugged = bat is None or bat.power_plugged
            if plugged and (t is None or t <= cool):
                log(f"devam ({waited:.0f} sn bekledi, GPU {t} C)")
                break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--init", default=os.path.join(ROOT, "weights", "rt4ksr", "rt4ksr_x2.pth"))
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--ckpt-min", type=float, default=3.0)
    ap.add_argument("--hot", type=float, default=88.0)
    ap.add_argument("--cool", type=float, default=78.0)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--val-shards", type=int, default=4)
    args = ap.parse_args()
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    run_dir = os.path.join(ROOT, "runs", f"train_{args.name}")
    ck_dir = os.path.join(ROOT, "weights", "ft", args.name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ck_dir, exist_ok=True)
    last_path = os.path.join(ck_dir, "last.pt")
    best_rel = os.path.join(ROOT, "weights", "rt4ksr", f"{args.name}_best.pth")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(os.path.join(run_dir, "olaylar.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    net = RT4KSR(upscale=2, rep=False).to(device)
    ck = torch.load(args.init, map_location="cpu", weights_only=True)
    net.load_state_dict(clean_checkpoint(ck["state_dict"]), strict=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=args.lr * 0.01)
    it, best = 0, -1.0
    if os.path.exists(last_path):
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        it, best = st["it"], st["best"]
        log(f"kaldigi yerden devam: iter {it}, en iyi {best:.3f}")

    train_files = shards(args.train)
    val_files = shards(args.val)[: args.val_shards]
    if not train_files or not val_files:
        raise SystemExit("egitim ya da dogrulama parcasi yok")
    val = [load_shard(p) for p in val_files]
    stream = ShardStream(train_files, args.batch, args.group, seed=it + 7)
    gpu = GpuMonitor(10.0)
    gpu.start()
    time.sleep(1.5)

    new_log = not os.path.exists(os.path.join(run_dir, "log.csv"))
    logf = open(os.path.join(run_dir, "log.csv"), "a", newline="", encoding="utf-8")
    w = csv.writer(logf)
    if new_log:
        w.writerow(["iter", "zaman", "kayip", "lr", "val_psnr", "val_bicubic", "sicaklik_c", "guc_w", "it_sn"])
        base = evaluate(net, val, device)
        log(f"baslangic dogrulama: hazir RT4KSR {base['psnr']:.3f} dB, bicubic {base['bicubic']:.3f} dB, "
            f"{sum(len(v[0]) for v in val)} yama, egitim {len(train_files)} parca")
        w.writerow([0, time.strftime("%H:%M:%S"), "", args.lr, round(base["psnr"], 4), round(base["bicubic"], 4),
                    gpu.last.get("sicaklik_c"), gpu.last.get("guc_w"), ""])
        best = max(best, base["psnr"])

    net.train()
    t_ck = time.time()
    t_win, loss_sum, n_sum = time.time(), 0.0, 0
    try:
        while it < args.iters:
            if it % 50 == 0:
                waited = wait_safe(gpu, args.hot, args.cool, log)
                t_win += waited
            lr_b, hr_b = stream.next(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sr = net(lr_b)
            loss = F.l1_loss(sr.float(), hr_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            loss_sum += float(loss.detach())
            n_sum += 1
            if it % args.eval_every == 0 or it == args.iters:
                ev = evaluate(net, val, device)
                dt = time.time() - t_win
                w.writerow([it, time.strftime("%H:%M:%S"), round(loss_sum / n_sum, 5), f"{sched.get_last_lr()[0]:.2e}",
                            round(ev["psnr"], 4), round(ev["bicubic"], 4), gpu.last.get("sicaklik_c"),
                            gpu.last.get("guc_w"), round(n_sum / dt, 1)])
                logf.flush()
                mark = ""
                if ev["psnr"] > best:
                    best = ev["psnr"]
                    save_release(net, best_rel)
                    mark = " (en iyi, kaydedildi)"
                log(f"iter {it}: kayip {loss_sum / n_sum:.4f}, dogrulama {ev['psnr']:.3f} dB{mark}, "
                    f"{n_sum / dt:.1f} it/sn, GPU {gpu.last.get('sicaklik_c')} C")
                t_win, loss_sum, n_sum = time.time(), 0.0, 0
            if time.time() - t_ck > args.ckpt_min * 60 or it == args.iters:
                tmp = last_path + ".tmp"
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "it": it, "best": best}, tmp)
                os.replace(tmp, last_path)
                t_ck = time.time()
    finally:
        tmp = last_path + ".tmp"
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "it": it, "best": best}, tmp)
        os.replace(tmp, last_path)
        logf.close()
        gpu.stop()
        log(f"durdu: iter {it}, en iyi dogrulama {best:.3f} dB")


if __name__ == "__main__":
    main()
