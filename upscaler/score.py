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
    """GPU'da PSNR + SSIM (Y, 0-255). SSIM: 11x11 Gauss, ayrik evrisim. Girdi bitisik olmali
    (kirpilmis gorunumde evrisim 4 kat yavas: 54 -> 13 ms, 4K)."""

    def __init__(self, device: str = "cuda") -> None:
        g = _gauss(device=device)
        self.kx = g.view(1, 1, 1, -1)
        self.ky = g.view(1, 1, -1, 1)
        self.c1 = (0.01 * 255) ** 2
        self.c2 = (0.03 * 255) ** 2

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.conv2d(x, self.kx), self.ky)

    @torch.no_grad()
    def tensors(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(psnr, ssim) 0 boyutlu tensorler (senkronizasyon yok)."""
        mse = torch.mean((a - b) ** 2).clamp_min(1e-10)
        psnr = 10.0 * torch.log10(255.0 ** 2 / mse)
        x = a[None, None]
        y = b[None, None]
        mx, my = self._blur(x), self._blur(y)
        sxx = self._blur(x * x) - mx * mx
        syy = self._blur(y * y) - my * my
        sxy = self._blur(x * y) - mx * my
        ssim = ((2 * mx * my + self.c1) * (2 * sxy + self.c2)) / ((mx * mx + my * my + self.c1) * (sxx + syy + self.c2))
        return psnr, ssim.mean()

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
        p, s = self.tensors(a.contiguous(), b.contiguous())
        return float(p.item()), float(s.item())


CROP = (slice(EXCLUDE_TOP_4K, -BORDER_4K), slice(BORDER_4K, -BORDER_4K))


def ssim_psnr_cpu(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Y (float32, 0-255): PSNR ve SSIM (11x11 Gauss, sigma 1.5, gecerli bolge; Wang vd. 2004)."""
    import cv2
    mse = float(np.mean((x - y) ** 2))
    psnr = 99.0 if mse <= 1e-10 else 10.0 * float(np.log10(255.0 ** 2 / mse))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def bl(z):
        return cv2.GaussianBlur(z, (11, 11), 1.5)[5:-5, 5:-5]
    mx, my = bl(x), bl(y)
    sxx = bl(x * x) - mx * mx
    syy = bl(y * y) - my * my
    sxy = bl(x * y) - mx * my
    ssim = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return psnr, float(ssim.mean())


class GtBank:
    """GT segmentinin secilen karelerinin Y kanali (RAM, uint8, puan bolgesine kirpilmis)."""

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
                # RAM'de (uint8, kirpilmis ve bitisik): puan CPU'da hesaplanir, VRAM hatta kalir
                self.frames[g] = np.ascontiguousarray(y.add_(0.5).clamp_(0, 255).to(torch.uint8)[CROP].cpu().numpy())
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
    """Cikis dongusunden cagrilir. Y ana akista hesaplanip pinned bellege asenkron kopyalanir (~1 ms);
    PSNR/SSIM arka plan is parcaciginda CPU'da (cv2, 2 is parcacigi) hesaplanir. GPU'da SSIM
    TensorRT islerini bekletip islem p99'u 70 ms'ye cikariyordu (2026-09-16). Ayni anda tek is."""

    def __init__(self, bank: GtBank, max_per_s: float = 2.0) -> None:
        import threading

        import cv2
        cv2.setNumThreads(2)
        self.bank = bank
        self.interval = 1.0 / max_per_s
        self._last_t = -1e9
        self._want = "gercek"  # siniflar sirayla puanlanir
        self.rows: list[tuple] = []  # (t, g, sinif, alpha, psnr, ssim)
        self.skipped_rate = 0
        self.skipped_busy = 0
        self.skipped_budget = 0
        self.not_grid = 0       # icerik konumu GT izgarasina dusmuyor (hizalama)
        self.not_in_bank = 0    # izgarada ama bankada tutulmayan kare (bellek icin atlanan)
        self.no_barcode = 0
        self.gap = 0
        self.copy_ms: list[float] = []
        self.cpu_ms: list[float] = []
        h = 2160 - EXCLUDE_TOP_4K - BORDER_4K
        w = 3840 - 2 * BORDER_4K
        self._pinned = torch.empty((h, w), dtype=torch.float32).pin_memory()
        self._gpu = torch.empty((2160, 3840), dtype=torch.float32, device="cuda")
        self._job: tuple | None = None
        self._busy = threading.Event()
        self._wake = threading.Event()
        self._stop = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

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

    def _worker(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                return
            item, ev = self._job
            ev.synchronize()  # sadece bu is parcacigini bekletir
            t0 = time.perf_counter()
            x = self._pinned.numpy()
            gt = self.bank.frames[item[1]].astype(np.float32)
            psnr, ssim = ssim_psnr_cpu(x, gt)
            with self._lock:
                self.rows.append(item + (round(psnr, 3), round(ssim, 5)))
                self.cpu_ms.append((time.perf_counter() - t0) * 1000)
            self._job = None
            self._busy.clear()

    def offer(self, y: torch.Tensor, ia, ib, alpha: float, t: float, budget_s: float) -> None:
        """y: (1,3,2160,3840) BGR cikis. budget_s: bir sonraki tike kalan sure."""
        pos = self.content_pos(ia, ib, alpha)
        if pos is None:
            if ia is None or (ib is None and alpha > 1e-3):
                self.no_barcode += 1
            else:
                self.gap += 1
            return
        gf = pos * self.bank.ratio
        if abs(gf - round(gf)) > 0.02:
            self.not_grid += 1
            return
        g = self.bank.gt_index(pos)
        if g is None:
            self.not_in_bank += 1
            return
        real = alpha <= 1e-3 or alpha >= 1 - 1e-3
        cls = "gercek" if real else "ara"
        gap = t - self._last_t
        if gap < self.interval or (cls != self._want and gap < 2 * self.interval):
            self.skipped_rate += 1
            return
        if self._busy.is_set():
            self.skipped_busy += 1
            return
        if budget_s < 0.004:
            self.skipped_budget += 1
            return
        self._last_t = t
        self._want = "ara" if cls == "gercek" else "gercek"
        t0 = time.perf_counter()
        yb = y[0]
        torch.add(torch.add(yb[0].float().mul_(0.0722), yb[1].float(), alpha=0.7152), yb[2].float(), alpha=0.2126,
                  out=self._gpu)
        self._pinned.copy_(self._gpu[CROP], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._busy.set()
        self._job = ((round(t, 3), g, cls, round(alpha, 3)), ev)
        self._wake.set()
        self.copy_ms.append((time.perf_counter() - t0) * 1000)

    def idle(self, budget_s: float, t: float | None = None) -> None:
        pass

    def finish(self) -> None:
        t0 = time.perf_counter()
        while self._busy.is_set() and time.perf_counter() - t0 < 5:
            time.sleep(0.01)
        self._stop = True
        self._wake.set()

    def summary(self) -> dict:
        out = {"gt": self.bank.meta["gt"], "gt_start_s": self.bank.meta["gt_start_s"], "kaydirma": self.bank.shift,
               "gt_yukleme_sn": round(self.bank.load_s, 1), "gt_kare": len(self.bank.frames),
               "puanlanan": len(self.rows), "oran_atlanan": self.skipped_rate, "mesgul_atlanan": self.skipped_busy,
               "butce_atlanan": self.skipped_budget, "izgara_disi": self.not_grid, "bankada_yok": self.not_in_bank,
               "serit_okunamayan": self.no_barcode, "bosluk": self.gap,
               "y_kopya_ms_p50_p95": [round(float(np.percentile(self.copy_ms, q)), 2) for q in (50, 95)]
               if self.copy_ms else None,
               "cpu_puan_ms_p50": round(float(np.median(self.cpu_ms)), 1) if self.cpu_ms else None}
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


