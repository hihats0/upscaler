"""Sikistirma onarim agi v0 egitimi (F27, goal 2026-09-19 B).

Veri: tools/make_pairs.py --target 1080 parcalari. lr = 4,8 Mbps H.264 1080p yama, hr = ayni
karenin sikistirilmamis 1080p yamasi (ayni filtre zinciri). Sadece data/clips CC klipleri.
Kayip: L1 + genlik spektrumu (--amp-weight). GAN yok.
Dogrulama (sabit, TOD benzeri, egitimden ayri klipler): cikisin ve girdinin hedefe PSNR-Y'si,
ve hf = cikisin yuksek frekans enerjisi / hedefinki (Nyquist'in yarisi ustu). Girdinin hf'si de
yazilir: cikis hf'si girdininkinden dusukse ag detay siliyor demektir.

Guvenlik: GPU >= --hot C'de <= --cool C'ye kadar duraklama, fis cekilirse duraklama (train_sr).
Kayit: runs/train_<ad>/log.csv, en iyi PSNR weights/repair/<ad>.pth.

    .venv/Scripts/python.exe tools/train_repair.py --name rep_v0 --train rep_train --val rep_val --ch 32 --blocks 4
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from train_sr import ShardStream, fft_amp_loss, load_shard, psnr_y, shards, wait_safe  # noqa: E402
from upscaler.models.repair import RepairNet, save_repair  # noqa: E402
from upscaler.watch import GpuMonitor  # noqa: E402

Y_COEF = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)


def hf_energy(x: torch.Tensor) -> torch.Tensor:
    """1080p yamada Y kanalinin Nyquist'in yarisi ustundeki enerjisi (ince doku, kenar)."""
    y = (x.clamp(0, 1) * Y_COEF.to(x.device)).sum(1)
    y = y - y.mean((1, 2), keepdim=True)
    p = torch.fft.rfft2(y, norm="ortho").abs() ** 2
    h, w = y.shape[-2:]
    fy = torch.fft.fftfreq(h, device=x.device).abs()[:, None] * 2
    fx = torch.fft.rfftfreq(w, device=x.device)[None, :] * 2
    return (p * (torch.maximum(fy, fx) >= 0.5).float()).sum((1, 2))


@torch.no_grad()
def evaluate(net, val, device: str) -> dict:
    net.eval()
    po, pi = [], []
    e_out = e_in = e_hr = 0.0
    for lr_np, hr_np in val:
        for s in range(0, len(lr_np), 64):
            lr = torch.from_numpy(lr_np[s:s + 64]).to(device).permute(0, 3, 1, 2).float() / 255
            hr = torch.from_numpy(hr_np[s:s + 64]).to(device).permute(0, 3, 1, 2).float() / 255
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(lr).float()
            out = (out.clamp(0, 1) * 255).round() / 255  # canli hat gibi 8 bit
            po.append(psnr_y(out, hr))
            pi.append(psnr_y(lr, hr))
            e_out += float(hf_energy(out).sum())
            e_in += float(hf_energy(lr).sum())
            e_hr += float(hf_energy(hr).sum())
    net.train()
    return {"psnr": float(torch.cat(po).mean()), "psnr_in": float(torch.cat(pi).mean()),
            "hf": e_out / max(e_hr, 1e-12), "hf_in": e_in / max(e_hr, 1e-12)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--amp-weight", type=float, default=1.0, help="genlik spektrumu kaybi agirligi")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--hot", type=float, default=80.0)
    ap.add_argument("--cool", type=float, default=70.0)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--val-shards", type=int, default=6)
    args = ap.parse_args()
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    run_dir = os.path.join(ROOT, "runs", f"train_{args.name}")
    os.makedirs(run_dir, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(os.path.join(run_dir, "olaylar.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    net = RepairNet(args.ch, args.blocks).to(device)
    ema = copy.deepcopy(net).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=args.lr * 0.01)
    train_files, val_files = shards(args.train), shards(args.val)[: args.val_shards]
    if not train_files or not val_files:
        raise SystemExit("egitim ya da dogrulama parcasi yok")
    val = [load_shard(p) for p in val_files]
    stream = ShardStream(train_files, args.batch, args.group, seed=7)
    gpu = GpuMonitor(10.0)
    gpu.start()
    time.sleep(1.5)
    logf = open(os.path.join(run_dir, "log.csv"), "w", newline="", encoding="utf-8")
    w = csv.writer(logf)
    w.writerow(["iter", "zaman", "kayip", "lr", "val_psnr", "val_psnr_girdi", "val_hf", "val_hf_girdi",
                "sicaklik_c", "guc_w", "it_sn"])
    base = evaluate(net, val, device)
    log(f"baslangic: girdi {base['psnr_in']:.3f} dB (hf {base['hf_in']:.3f}), ag {base['psnr']:.3f} dB, "
        f"{sum(len(v[0]) for v in val)} dogrulama yamasi, {len(train_files)} egitim parcasi, "
        f"param {sum(p.numel() for p in net.parameters()) / 1000:.1f}k")
    best = -1.0
    it = 0
    t_win, loss_sum, n_sum = time.time(), 0.0, 0
    try:
        while it < args.iters:
            if it % 50 == 0:
                t_win += wait_safe(gpu, args.hot, args.cool, log)
            lr_b, hr_b = stream.next(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(lr_b)
            out = out.float()
            loss = F.l1_loss(out, hr_b)
            if args.amp_weight > 0:
                loss = loss + args.amp_weight * fft_amp_loss(out, hr_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(pn, 1 - args.ema)
            it += 1
            loss_sum += float(loss.detach())
            n_sum += 1
            if it % args.eval_every == 0 or it == args.iters:
                ev = evaluate(ema, val, device)
                dt = time.time() - t_win
                w.writerow([it, time.strftime("%H:%M:%S"), round(loss_sum / n_sum, 5), f"{sched.get_last_lr()[0]:.2e}",
                            round(ev["psnr"], 4), round(ev["psnr_in"], 4), round(ev["hf"], 4), round(ev["hf_in"], 4),
                            gpu.last.get("sicaklik_c"), gpu.last.get("guc_w"), round(n_sum / dt, 1)])
                logf.flush()
                mark = ""
                if ev["psnr"] > best:
                    best = ev["psnr"]
                    save_repair(ema, args.name)
                    mark = " (en iyi, kaydedildi)"
                log(f"iter {it}: kayip {loss_sum / n_sum:.4f}, dogrulama {ev['psnr']:.3f} dB "
                    f"(girdi {ev['psnr_in']:.3f}, fark {ev['psnr'] - ev['psnr_in']:+.3f}), hf {ev['hf']:.3f} "
                    f"(girdi {ev['hf_in']:.3f}){mark}, {n_sum / dt:.1f} it/sn, GPU {gpu.last.get('sicaklik_c')} C")
                t_win, loss_sum, n_sum = time.time(), 0.0, 0
    finally:
        logf.close()
        gpu.stop()
        log(f"durdu: iter {it}, en iyi dogrulama {best:.3f} dB")


if __name__ == "__main__":
    main()
