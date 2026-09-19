"""AI yeniden cizim agi ince ayari (goal 2026-09-20, Asama C).

Veri: tools/make_pairs.py --target 1080sharp --temporal. lr = TOD benzeri 1080p yama, kareler
t-1, t, t+1, t+2 [N,4,P,P,3]; hr = ayni anin 4K karesinden KESKIN 1080p, kareler t, t+1 [N,2,P,P,3].
Sadece data/clips CC klipleri; dogrulama klipleri (5bgF-5I2P_M, O3gD6n0zoik) egitimde yok.

Ag (upscaler/models/ai.py AiNet): 1080p 2x2 katlanir, 540p'de conv+PReLU govde, girdiye artik eklenir.
frames=3 ise t-1, t, t+1 birlikte girer (1,5 sn gecikme gelecek kareye izin veriyor).

Kayip: L1 + LPIPS (VGG, algisal) + GAN (PatchDisc, hinge) + zamansal tutarlilik:
|(out_t+1 - out_t) - (hr_t+1 - hr_t)|. Once --warm adim sadece L1 + LPIPS (GAN yok), sonra GAN.
EMA 0,999. En iyi = dogrulamada en dusuk LPIPS (PSNR ve hf de yazilir).

Guvenlik: GPU >= --hot C'de <= --cool C'ye kadar duraklama, fis cekilirse duraklama.

    .venv/Scripts/python.exe tools/train_ai.py --name ai_v1 --train ai_train_a ai_train_b --val ai_val --feat 48 --conv 8 --frames 3
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from train_repair import hf_energy  # noqa: E402
from train_sr import PatchDisc, psnr_y, shards, wait_safe  # noqa: E402
from upscaler.models.ai import AiNet, load_ai, save_ai  # noqa: E402
from upscaler.watch import GpuMonitor  # noqa: E402


def watch_running() -> bool:
    """Yigit mac izliyor mu (python -m upscaler watch)? Izlerken egitim GPU'yu bosaltir."""
    import psutil
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            args = p.info["cmdline"] or []
        except Exception:  # noqa: BLE001
            continue
        # watch.bat: python.exe -m upscaler watch ...
        if any(args[i:i + 3] == ["-m", "upscaler", "watch"] for i in range(len(args) - 2)):
            return True
    return False


def wait_watch(log) -> float:
    waited = 0.0
    if watch_running():
        log("duraklama: watch acik (Yigit izliyor)")
        while watch_running():
            time.sleep(10)
            waited += 10
        log(f"devam: watch kapandi ({waited:.0f} sn)")
    return waited


def load_shard(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["lr"], z["hr"]


class TemporalStream:
    """[N,4,P,P,3] / [N,2,P,P,3] parcalardan karistirilmis yigin (RAM'de en fazla `group` parca)."""

    def __init__(self, files: list[str], batch: int, group: int, seed: int) -> None:
        self.files, self.batch, self.group = files, batch, group
        self.rng = np.random.default_rng(seed)
        self._idx = np.zeros(0, int)
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
        lr = torch.from_numpy(self._lr[ii]).to(device, non_blocking=True)  # N,4,P,P,3
        hr = torch.from_numpy(self._hr[ii]).to(device, non_blocking=True)  # N,2,P,P,3
        lr = lr.permute(0, 1, 4, 2, 3).float() / 255  # N,4,3,P,P
        hr = hr.permute(0, 1, 4, 2, 3).float() / 255
        k = int(self.rng.integers(4))
        if k:
            lr, hr = torch.rot90(lr, k, (3, 4)), torch.rot90(hr, k, (3, 4))
        if self.rng.random() < 0.5:
            lr, hr = lr.flip(4), hr.flip(4)
        if self.rng.random() < 0.5:  # zamani ters cevir: t, t+1 -> t+1, t (pencere simetrik kalir)
            lr, hr = lr.flip(1), hr.flip(1)
        return lr.contiguous(), hr.contiguous()


def net_inputs(lr: torch.Tensor, frames: int) -> torch.Tensor:
    """lr [N,4,3,P,P] (t-1, t, t+1, t+2) -> [2N, 3*frames, P, P]: once t'nin, sonra t+1'in girdisi."""
    n, _, c, p, q = lr.shape
    if frames == 1:
        x = lr[:, 1:3]
    elif frames == 3:
        x = torch.stack([lr[:, 0:3], lr[:, 1:4]], 1)  # N,2,3,3,P,P
    else:
        raise ValueError("frames 1 ya da 3")
    return x.transpose(0, 1).reshape(2 * n, -1, p, q)


@torch.no_grad()
def evaluate(net, val, frames: int, lp, device: str) -> dict:
    net.eval()
    po, pi, lps, lpi = [], [], [], []
    e_out = e_in = e_hr = 0.0
    tfl_o = tfl_g = 0.0
    for lr_np, hr_np in val:
        for s in range(0, len(lr_np), 32):
            lr = torch.from_numpy(lr_np[s:s + 32]).to(device).permute(0, 1, 4, 2, 3).float() / 255
            hr = torch.from_numpy(hr_np[s:s + 32]).to(device).permute(0, 1, 4, 2, 3).float() / 255
            n = lr.shape[0]
            x = net_inputs(lr, frames)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x).float()
            out = (out.clamp(0, 1) * 255).round() / 255
            o0, o1 = out[:n], out[n:]
            h0, h1 = hr[:, 0], hr[:, 1]
            i0 = lr[:, 1]
            po.append(psnr_y(o0, h0))
            pi.append(psnr_y(i0, h0))
            lps.append(lp(o0 * 2 - 1, h0 * 2 - 1).flatten())
            lpi.append(lp(i0 * 2 - 1, h0 * 2 - 1).flatten())
            e_out += float(hf_energy(o0).sum())
            e_in += float(hf_energy(i0).sum())
            e_hr += float(hf_energy(h0).sum())
            # titreme vekili: hedefin durgun oldugu piksellerde ardisik kare farki
            dg = (h1 - h0).abs().mean(1, keepdim=True)
            m = (F.avg_pool2d(dg, 7, 1, 3) < 1.0 / 255).float()
            tfl_o += float(((o1 - o0).abs().mean(1, keepdim=True) * m).sum())
            tfl_g += float((dg * m).sum())
    net.train()
    return {"psnr": float(torch.cat(po).mean()), "psnr_in": float(torch.cat(pi).mean()),
            "lpips": float(torch.cat(lps).mean()), "lpips_in": float(torch.cat(lpi).mean()),
            "hf": e_out / max(e_hr, 1e-12), "hf_in": e_in / max(e_hr, 1e-12),
            "titreme": tfl_o / max(tfl_g, 1e-12)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--feat", type=int, default=48)
    ap.add_argument("--conv", type=int, default=8)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--init", default="", help="weights/ai/<ad>.pth'den baslat")
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--warm", type=int, default=5000, help="GAN'siz ilk adimlar")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--l1", type=float, default=1.0)
    ap.add_argument("--lpips", type=float, default=1.0)
    ap.add_argument("--gan", type=float, default=0.05)
    ap.add_argument("--temporal", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--hot", type=float, default=80.0)
    ap.add_argument("--cool", type=float, default=70.0)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--disc-init", default="", help="runs/train_<ad>/disc.pth: GAN hakemini kaldigi yerden al")
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

    import lpips
    lp_train = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    lp_eval = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    for m in (lp_train, lp_eval):
        for p_ in m.parameters():
            p_.requires_grad_(False)
    if args.init:
        net = load_ai(args.init).to(device)
        args.feat, args.conv, args.frames = net.feat, net.conv, net.frames
    else:
        net = AiNet(args.feat, args.conv, args.frames).to(device)
    ema = copy.deepcopy(net).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    disc = PatchDisc(48).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.9, 0.99))
    if args.disc_init:
        dk = torch.load(args.disc_init, map_location=device, weights_only=True)
        disc.load_state_dict(dk["disc"])
        opt_d.load_state_dict(dk["opt_d"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=args.lr * 0.05)
    train_files = sum((shards(t) for t in args.train), [])
    val = [load_shard(p) for p in shards(args.val)]
    if not train_files or not val:
        raise SystemExit("egitim ya da dogrulama parcasi yok")
    stream = TemporalStream(train_files, args.batch, args.group, seed=11)
    gpu = GpuMonitor(10.0)
    gpu.start()
    time.sleep(1.5)
    logf = open(os.path.join(run_dir, "log.csv"), "w", newline="", encoding="utf-8")
    w = csv.writer(logf)
    w.writerow(["iter", "zaman", "g_kayip", "d_kayip", "lr", "psnr", "psnr_girdi", "lpips", "lpips_girdi",
                "hf", "hf_girdi", "titreme", "sicaklik_c", "guc_w", "it_sn"])
    base = evaluate(ema, val, args.frames, lp_eval, device)
    log(f"baslangic ({args}): girdi {base['psnr_in']:.3f} dB LPIPS {base['lpips_in']:.4f} hf {base['hf_in']:.3f}; "
        f"ag {base['psnr']:.3f} dB LPIPS {base['lpips']:.4f} hf {base['hf']:.3f} titreme {base['titreme']:.3f}; "
        f"{sum(len(v[0]) for v in val)} dogrulama, {len(train_files)} egitim parcasi, "
        f"param {sum(p.numel() for p in net.parameters()) / 1000:.1f}k")
    best = 9.0
    it = 0
    t_watch = 0.0
    t_win, gl_sum, dl_sum, n_sum = time.time(), 0.0, 0.0, 0
    try:
        while it < args.iters:
            if it % 50 == 0:
                t_win += wait_safe(gpu, args.hot, args.cool, log)
            if time.time() - t_watch > 5.0:  # Yigit watch acarsa en gec 5 sn icinde GPU'yu birak
                t_win += wait_watch(log)
                t_watch = time.time()
            lr_b, hr_b = stream.next(device)
            n = lr_b.shape[0]
            x = net_inputs(lr_b, args.frames)
            tgt = hr_b.transpose(0, 1).reshape(2 * n, 3, *hr_b.shape[-2:])  # t'ler, sonra t+1'ler
            use_gan = args.gan > 0 and it >= args.warm
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x)
            out = out.float()
            loss = args.l1 * F.l1_loss(out, tgt)
            if args.lpips > 0:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = loss + args.lpips * lp_train(out * 2 - 1, tgt * 2 - 1).float().mean()
            if args.temporal > 0:
                loss = loss + args.temporal * F.l1_loss(out[n:] - out[:n], tgt[n:] - tgt[:n])
            if use_gan:
                for p_ in disc.parameters():
                    p_.requires_grad_(False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = loss - args.gan * disc(out).float().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            d_loss = torch.zeros(())
            if args.gan > 0:  # hakem isinmada da egitilir: GAN acildiginda hazir olsun
                for p_ in disc.parameters():
                    p_.requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    d_real = disc(tgt).float()
                    d_fake = disc(out.detach()).float()
                d_loss = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(pn, 1 - args.ema)
            it += 1
            gl_sum += float(loss.detach())
            dl_sum += float(d_loss.detach())
            n_sum += 1
            if it % args.eval_every == 0 or it == args.iters:
                ev = evaluate(ema, val, args.frames, lp_eval, device)
                dt = time.time() - t_win
                w.writerow([it, time.strftime("%H:%M:%S"), round(gl_sum / n_sum, 5), round(dl_sum / n_sum, 4),
                            f"{sched.get_last_lr()[0]:.2e}", round(ev["psnr"], 4), round(ev["psnr_in"], 4),
                            round(ev["lpips"], 4), round(ev["lpips_in"], 4), round(ev["hf"], 4), round(ev["hf_in"], 4),
                            round(ev["titreme"], 4), gpu.last.get("sicaklik_c"), gpu.last.get("guc_w"),
                            round(n_sum / dt, 1)])
                logf.flush()
                save_ai(ema, args.name + "_son", {"iter": it})
                torch.save({"disc": disc.state_dict(), "opt_d": opt_d.state_dict(), "iter": it},
                           os.path.join(run_dir, "disc.pth"))
                mark = ""
                if ev["lpips"] < best:
                    best = ev["lpips"]
                    save_ai(ema, args.name, {"iter": it, "eval": ev})
                    mark = " (en iyi, kaydedildi)"
                log(f"iter {it}: g {gl_sum / n_sum:.4f} d {dl_sum / n_sum:.3f} | {ev['psnr']:.3f} dB "
                    f"(girdi {ev['psnr_in']:.3f}) LPIPS {ev['lpips']:.4f} (girdi {ev['lpips_in']:.4f}) "
                    f"hf {ev['hf']:.3f} (girdi {ev['hf_in']:.3f}) titreme {ev['titreme']:.3f}{mark}, "
                    f"{n_sum / dt:.1f} it/sn, GPU {gpu.last.get('sicaklik_c')} C")
                t_win, gl_sum, dl_sum, n_sum = time.time(), 0.0, 0.0, 0
    finally:
        logf.close()
        gpu.stop()
        log(f"durdu: iter {it}, en iyi LPIPS {best:.4f}")


if __name__ == "__main__":
    main()
