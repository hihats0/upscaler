"""Canli puan (Hat 1.3): simule TOD klibi canli hattan gecer, cikis GT 4K60 ile karsilastirilir.

Akis (tools/live_score.py):
  tools/tod_sim.py ile uretilen 1080p50 klip ffplay penceresinde doner -> WGC -> hat -> 4K cikis.
  Yakalanan her karenin klip kare numarasi ust seritten okunur (tod_sim.read_barcode) ve tampona
  meta olarak girer. Cikis tikinde gosterilen icerigin klip konumu i + alpha, GT karesi
  g = (gt_fps / sim_fps) * (i + alpha). g tam sayi ve bankadaysa Y kanalinda PSNR ve SSIM hesaplanir.

GT bankasi: GT segmenti ffmpeg ile cozulur, Y (BT.709, tam aralik) VRAM'de uint8 tutulur. Bellek
icin sadece (g - shift) % 6 in keep kareleri saklanir: 0 = gercek kare tiki, 3 = alpha 0,5 ara
kare (50->60'ta en zor ara kare). shift: kasitli kaydirma testi (GT N kare kaydirilir).

Puanlama sadece bir sonraki tike yeterli zaman varken yapilir (canli 60 FPS once gelir).
Ust serit (kare numarasi) ve kenar bosluklari puana girmez. Hicbir kare diske yazilmaz.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Kare numarasi seridi (tools/tod_sim.py uretir): ust STRIP_H satirda 24 hucre, CELL_W px.
BITS = 20
CHECK_BITS = 4
CELL_W = 64
STRIP_H = 16
EXCLUDE_TOP_4K = 48   # 1080p'de 16 satirlik serit -> 4K'da 32, SR yayilmasi icin pay
BORDER_4K = 8


def read_barcode(img: np.ndarray) -> int | None:
    """1080p kareden (BGRA ya da gri) klip kare numarasi: 20 bit + 4 bit denetim. Tutmazsa None."""
    if img.ndim == 3:
        img = img[..., 1]
    if img.shape[0] < STRIP_H or img.shape[1] < (BITS + CHECK_BITS + 2) * CELL_W:
        return None
    cells = img[STRIP_H // 2, CELL_W + CELL_W // 2:(BITS + CHECK_BITS + 1) * CELL_W:CELL_W]
    bits = np.asarray(cells[: BITS + CHECK_BITS]) > 128
    idx = 0
    for b in range(BITS):
        idx |= int(bits[b]) << b
    chk = 0
    for c in range(CHECK_BITS):
        chk |= int(bits[BITS + c]) << c
    if int(bits[:BITS].sum()) % 16 != chk:
        return None
    return idx


def _gauss(size: int = 11, sigma: float = 1.5, device="cuda") -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def luma_bgr(y: torch.Tensor) -> torch.Tensor:
    """(1,3,H,W) BGR (uint8 ya da float 0-255) -> (H,W) float32 BT.709 Y."""
    x = y[0].float()
    return 0.0722 * x[0] + 0.7152 * x[1] + 0.2126 * x[2]


class YMetric:
    """GPU'da PSNR + SSIM (Y, 0-255). SSIM: 11x11 Gauss, ayrik evrisim."""

    def __init__(self, device: str = "cuda") -> None:
        g = _gauss(device=device)
        self.kx = g.view(1, 1, 1, -1)
        self.ky = g.view(1, 1, -1, 1)
        self.c1 = (0.01 * 255) ** 2
        self.c2 = (0.03 * 255) ** 2

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.conv2d(x, self.kx), self.ky)

    @torch.no_grad()
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
        mse = torch.mean((a - b) ** 2).item()
        psnr = 99.0 if mse <= 1e-10 else 10.0 * float(np.log10(255.0 ** 2 / mse))
        x = a[None, None]
        y = b[None, None]
        mx, my = self._blur(x), self._blur(y)
        sxx = self._blur(x * x) - mx * mx
        syy = self._blur(y * y) - my * my
        sxy = self._blur(x * y) - mx * my
        ssim = ((2 * mx * my + self.c1) * (2 * sxy + self.c2)) / ((mx * mx + my * my + self.c1) * (sxx + syy + self.c2))
        return psnr, float(ssim.mean().item())


class GtBank:
    """GT segmentinin secilen karelerinin Y kanali (VRAM, uint8)."""

    def __init__(self, meta_path: str, shift: int = 0, keep: tuple[int, ...] = (0, 3), period: int = 6,
                 device: str = "cuda") -> None:
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.shift, self.keep, self.period = shift, keep, period
        self.ratio = self.meta["gt_fps"] / self.meta["sim_fps"]
        self.frames: dict[int, torch.Tensor] = {}
        self.device = device
        self.load_s = 0.0

    def wanted(self, g: int) -> bool:
        return (g - self.shift) % self.period in self.keep

    def load(self) -> "GtBank":
        t0 = time.perf_counter()
        m = self.meta
        gt = os.path.join(ROOT, m["gt"])
        w, h = 3840, 2160
        vf = "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(m["gt_start_s"]), "-t", str(m["seconds"]),
               "-i", gt, "-an", "-vf", vf, "-f", "rawvideo", "pipe:1"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        n = w * h * 3
        coef = torch.tensor([0.2126, 0.7152, 0.0722], device=self.device).view(3, 1, 1)
        g = 0
        buf = bytearray(n)
        view = memoryview(buf)
        while True:
            got = 0
            while got < n:
                r = p.stdout.readinto(view[got:])
                if not r:
                    break
                got += r
            if got < n:
                break
            if self.wanted(g):
                t = torch.frombuffer(buf, dtype=torch.uint8).view(h, w, 3).to(self.device, non_blocking=False)
                y = (t.permute(2, 0, 1).float() * coef).sum(0)
                self.frames[g] = y.add_(0.5).clamp_(0, 255).to(torch.uint8)
            g += 1
        p.wait()
        self.total = g
        self.load_s = time.perf_counter() - t0
        return self

    def gt_index(self, pos: float) -> int | None:
        """Klip konumu (kaynak kare biriminde) -> kaydirmali GT karesi (bankada yoksa None)."""
        gf = pos * self.ratio
        gi = round(gf)
        if abs(gf - gi) > 0.02:
            return None
        gi += self.shift
        return gi if gi in self.frames else None


class LiveScorer:
    """Cikis dongusunden cagrilir. Bosluk varsa hemen puanlar, yoksa tek kareyi bekletir."""

    def __init__(self, bank: GtBank, max_per_s: float = 4.0) -> None:
        self.bank = bank
        self.interval = 1.0 / max_per_s  # 4K SSIM pahali: saniyede birkac kare yeter
        self._last_t = -1e9
        self._want = "gercek"  # siniflar sirayla puanlanir
        self.skipped_rate = 0
        self.metric = YMetric()
        self.rows: list[tuple] = []  # (t, g, sinif, alpha, psnr, ssim)
        self.dropped_pending = 0
        self.skipped_nomatch = 0
        self.no_barcode = 0
        self.cost_ms: list[float] = []
        self._pending: tuple | None = None
        self._buf: torch.Tensor | None = None

    @staticmethod
    def content_pos(ia, ib, alpha: float) -> float | None:
        if ia is None:
            return None
        if alpha <= 1e-3:
            return float(ia)
        if ib is None:
            return None
        if alpha >= 1 - 1e-3:
            return float(ib)
        if ib <= ia or ib - ia > 3:  # dongu basi ya da uzun bosluk: ara kare anlamsiz
            return None
        return ia + alpha * (ib - ia)

    def offer(self, y: torch.Tensor, ia, ib, alpha: float, t: float, budget_s: float) -> None:
        """y: (1,3,2160,3840) BGR cikis. budget_s: bir sonraki tike kalan sure."""
        pos = self.content_pos(ia, ib, alpha)
        if pos is None:
            if ia is None:
                self.no_barcode += 1
            return
        g = self.bank.gt_index(pos)
        if g is None:
            self.skipped_nomatch += 1
            return
        real = alpha <= 1e-3 or alpha >= 1 - 1e-3
        cls = "gercek" if real else "ara"
        gap = t - self._last_t
        if gap < self.interval or (cls != self._want and gap < 2 * self.interval) or self._pending is not None:
            self.skipped_rate += 1
            return
        item = (t, g, cls, round(alpha, 3))
        est = (np.median(self.cost_ms[-50:]) / 1000 if self.cost_ms else 0.006) + 0.002
        self._last_t = t
        self._want = "ara" if cls == "gercek" else "gercek"
        if budget_s > est:
            self._score(luma_bgr(y), item)
        else:
            # Y'yi kopyala (ucuz), bir sonraki bos anda puanla.
            if self._buf is None:
                self._buf = torch.empty((y.shape[2], y.shape[3]), dtype=torch.float32, device=y.device)
            self._buf.copy_(luma_bgr(y))
            self._pending = item

    def idle(self, budget_s: float, t: float | None = None) -> None:
        if self._pending is None:
            return
        if t is not None and t - self._pending[0] > 1.0:  # hic bos an bulunamadi: birak
            self._pending = None
            self.dropped_pending += 1
            return
        est = (np.median(self.cost_ms[-50:]) / 1000 if self.cost_ms else 0.006) + 0.002
        if budget_s > est:
            item, self._pending = self._pending, None
            self._score(self._buf, item)

    def _score(self, yl: torch.Tensor, item: tuple) -> None:
        t0 = time.perf_counter()
        gt = self.bank.frames[item[1]]
        sl = (slice(EXCLUDE_TOP_4K, -BORDER_4K), slice(BORDER_4K, -BORDER_4K))
        psnr, ssim = self.metric(yl[sl], gt[sl].float())
        self.cost_ms.append((time.perf_counter() - t0) * 1000)
        self.rows.append(item + (round(psnr, 3), round(ssim, 5)))

    def summary(self) -> dict:
        out = {"gt": self.bank.meta["gt"], "gt_start_s": self.bank.meta["gt_start_s"], "kaydirma": self.bank.shift,
               "gt_yukleme_sn": round(self.bank.load_s, 1), "gt_kare": len(self.bank.frames),
               "puanlanan": len(self.rows), "oran_atlanan": self.skipped_rate, "bekleyen_birakilan": self.dropped_pending,
               "eslesmeyen": self.skipped_nomatch, "serit_okunamayan": self.no_barcode,
               "puan_maliyeti_ms_p50_p95": [round(float(np.percentile(self.cost_ms, q)), 2) for q in (50, 95)]
               if self.cost_ms else None}
        for cls in ("gercek", "ara"):
            r = [x for x in self.rows if x[2] == cls]
            if r:
                p = np.array([x[4] for x in r])
                s = np.array([x[5] for x in r])
                out[cls] = {"n": len(r), "psnr_ort": round(float(p.mean()), 3), "psnr_p5": round(float(np.percentile(p, 5)), 3),
                            "ssim_ort": round(float(s.mean()), 5)}
        return out

    def dump(self) -> list:
        return self.rows


