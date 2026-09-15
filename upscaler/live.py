"""Canli hat v0 (Hat 1.1 iskeleti).

Pencere (WGC) -> GPU halka tamponu + zaman cizgisi -> 60 Hz cikis dongusu
(sabit gecikme D) -> islemci (4K) -> onizleme (kucultulmus) ya da basliksiz.

Harici ekran yokken 4K kare GPU'da tam boyutta uretilir ve olculur; ekrana
kucultulmus onizleme gider. Hicbir kare diske yazilmaz.

Kullanim:
    .venv/Scripts/python.exe -m upscaler.live --title "TOD - Google Chrome" --delay 1.5 --seconds 30
    .venv/Scripts/python.exe -m upscaler.live --title upscaler-test-pattern --pattern --no-preview
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import winutil
from .capture import WgcSource
from .process import make_processor
from .ring import GpuFrameRing


@dataclass
class LiveConfig:
    title: str
    delay: float = 1.5
    seconds: float = 30.0
    out_fps: float = 60.0
    out_h: int = 2160
    out_w: int = 3840
    preview: bool = True
    pattern: bool = False
    min_interval_ms: int = 1
    report_every: float = 1.0
    quiet: bool = False
    proc: str = "baseline"  # baseline | rife | rife-lite
    dump_timing: str | None = None  # CSV yolu: benzersiz kare zamanlari (goruntu yok)
    snap: bool = True  # cikisi kaynak+cikis ortak izgarasina hizala


def _snap_grid(period: float | None, out_fps: float) -> tuple[float, int] | None:
    """Kaynak ve cikis hizi tam sayiysa ortak izgara: (adim sn, kaynak periyodundaki faz sayisi)."""
    if not period:
        return None
    src = 1.0 / period
    if abs(src - round(src)) > 1e-6 or abs(out_fps - round(out_fps)) > 1e-6:
        return None  # 29.97 gibi hizlarda hizalama yok
    lcm = math.lcm(round(src), round(out_fps))
    return 1.0 / lcm, round(lcm / src)


class Preview:
    """Kucuk pygame penceresi. 4K kareyi GPU'da kucultup gosterir."""

    def __init__(self, w: int = 960, h: int = 540, x: int = 900, y: int = 480) -> None:
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"
        import pygame
        self.pg = pygame
        pygame.init()
        self.w, self.h = w, h
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("upscaler-preview")
        self.surf = pygame.Surface((w, h))

    def show(self, bgr: torch.Tensor) -> None:
        small = F.interpolate(bgr.float(), size=(self.h, self.w), mode="area")
        rgb_xy = small[0].flip(0).permute(2, 1, 0).clamp_(0, 255).to(torch.uint8).cpu().numpy()
        self.pg.surfarray.blit_array(self.surf, rgb_xy)
        self.screen.blit(self.surf, (0, 0))
        self.pg.display.flip()
        for e in self.pg.event.get():
            if e.type == self.pg.QUIT:
                raise KeyboardInterrupt

    def close(self) -> None:
        self.pg.quit()


def _pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def run(cfg: LiveConfig) -> dict:
    winutil.dpi_aware()
    winutil.fine_timer()
    hwnd = winutil.find_window(cfg.title)

    capacity = math.ceil((cfg.delay + 1.0) * 60)
    ring = GpuFrameRing(capacity)
    proc = make_processor(cfg.proc, cfg.out_h, cfg.out_w)
    # Isinma: ilk CUDA cagrilari (cekirdek yukleme, bellek) dongude gec tik yaratmasin.
    for h, w in ((1080, 1920), (1020, 1920), (720, 1280)):
        z = torch.zeros((h, w, 4), dtype=torch.uint8, device="cuda")
        for _ in range(3):
            proc(z, z, 0.5)
    torch.cuda.synchronize()
    decode = None
    if cfg.pattern:
        from .testpattern import decode

    timing: list[tuple[float, int]] = []  # (raw, barkod): sadece sayi, tekrar oynatma icin

    def on_new_frame(raw: float, bgra: np.ndarray) -> None:
        meta = decode(bgra) if decode else None
        if cfg.dump_timing:
            timing.append((raw, -1 if meta is None else meta))
        ring.push(raw, bgra, meta)

    preview = Preview() if cfg.preview else None
    src = WgcSource(hwnd, on_new_frame, cfg.min_interval_ms).start()

    period = 1.0 / cfg.out_fps
    t0 = time.perf_counter()
    k = 0
    out = late = waiting = holds = 0
    proc_ms: list[float] = []
    buffer_ms: list[float] = []
    ticks: list[tuple[float, object, object, float]] = []  # desen dogrulamasi
    last_report = t0
    rep = dict(out=0, late=0, holds=0, unique=0)
    first_out: float | None = None
    last_out: float | None = None
    real_ticks = 0

    try:
        while True:
            T = t0 + k * period
            now = time.perf_counter()
            if now - t0 > cfg.seconds:
                break
            if src.error:
                raise src.error
            if now < T:
                if T - now > 0.002:
                    time.sleep(T - now - 0.0015)
                while time.perf_counter() < T:
                    pass
                now = time.perf_counter()
            if now - T > period:  # islemci yetismedi: kacan tikleri atla
                missed = int((now - T) / period)
                late += missed
                k += missed
                continue

            s = T - cfg.delay
            if ring.clock.period is None:
                # Saat kilitlenmeden cikis yok: isinma damgalari titrek, izgara belirsiz.
                waiting += 1
                k += 1
                continue
            grid = _snap_grid(ring.clock.period, cfg.out_fps) if cfg.snap else None
            if grid:
                # Kaynak zamanini kaynak+cikis ortak izgarasina oturt (50/60 -> 1/300 sn).
                # Boylece bazi tikler tam gercek kareye duser (ara kare yok) ve ara oranlar sabit.
                step, phases = grid
                origin = ring.clock._offset
                s = origin + round((s - origin) / step) * step
            p = ring.pick(s)
            if p is None:
                waiting += 1
                k += 1
                continue
            if grid:
                a = p.alpha * phases
                if abs(a - round(a)) < 0.15:  # kucuk ofset kaymasi oranı bozmasin
                    p.alpha = round(a) / phases
            real_ticks += p.alpha <= 1e-3 or p.alpha >= 1 - 1e-3
            try:
                t_a = time.perf_counter()
                fa, fb = p.frames()
                y = proc(fa, fb, p.alpha)
                if preview:
                    preview.show(y)
                else:
                    torch.cuda.synchronize()
                proc_ms.append((time.perf_counter() - t_a) * 1000)
            finally:
                p.release()
            out += 1
            if first_out is None:
                first_out = now
            last_out = now
            holds += p.hold
            buffer_ms.append((p.newest_t - s) * 1000)
            if cfg.pattern:
                ticks.append((s, p.a.meta, p.b.meta, p.alpha))
            k += 1

            if not cfg.quiet and now - last_report >= cfg.report_every:
                dt = now - last_report
                print(f"[{now - t0:5.1f}s] giris {(src.unique - rep['unique']) / dt:5.1f} fps | "
                      f"cikis {(out - rep['out']) / dt:5.1f} fps | gec {late - rep['late']:3d} | "
                      f"tutma {holds - rep['holds']:3d} | tampon {buffer_ms[-1]:6.0f} ms | "
                      f"islem p50 {_pct(proc_ms[-60:], 50):5.1f} ms | "
                      f"VRAM {torch.cuda.memory_allocated() / 2**30:4.2f} GB", flush=True)
                rep = dict(out=out, late=late, holds=holds, unique=src.unique)
                last_report = now
    finally:
        src.stop()
        if preview:
            preview.close()

    elapsed = time.perf_counter() - t0
    in_fps = src.unique_fps()
    if cfg.dump_timing:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.dump_timing)), exist_ok=True)
        with open(cfg.dump_timing, "w", encoding="utf-8") as f:
            f.write("raw_s,barcode\n")
            f.writelines(f"{r:.7f},{c}\n" for r, c in timing)
    summary = {
        "sure_sn": round(elapsed, 2),
        "kaynak": f"{ring.shape[1]}x{ring.shape[0]}" if ring.shape else None,
        "giris_fps": round(in_fps, 2) if in_fps else None,
        "saat_periyot_ms": round((ring.clock.period or 0) * 1000, 3),
        "cikis_fps_aktif": (round((out - 1) / (last_out - first_out), 2)
                            if out > 1 and last_out > first_out else None),
        "cikis_kare": out,
        "gec_tik": late,
        "bekleme_tik": waiting,
        "tutma_tik": holds,
        "islem_ms_p50_p95_max": [round(_pct(proc_ms, 50), 2), round(_pct(proc_ms, 95), 2),
                                 round(max(proc_ms), 2) if proc_ms else None],
        "tampon_ms_p5_p50": [round(_pct(buffer_ms, 5), 1), round(_pct(buffer_ms, 50), 1)],
        "vram_tepe_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "tampon_sifirlama": ring.resets,
        "yeniden_damgalama": ring.restamps,
        "tekrar_guncelleme": ring.repeats,
        "bosluk_doldurma": ring.hole_fills,
        "gercek_kare_tik_orani": round(real_ticks / out, 3) if out else None,
        "izgara_hizalama": cfg.snap,
        "islemci": proc.name,
    }
    if cfg.pattern:
        summary.update(_pattern_timeline_error(ticks, ring.clock.period or 0.02))
    return summary


def _pattern_timeline_error(ticks, period: float) -> dict:
    """Cikistaki her tik icin: secilen kaynak konumu (barkoddan) istenen zamana ne kadar yakin?

    tahmini_kaynak = idx_a + alpha * (idx_b - idx_a)
    s - period * tahmini_kaynak sabit olmali (sabit gecikme). Sapma = zaman hatasi.
    """
    resid, used = [], []
    for k, (s, ma, mb, alpha) in enumerate(ticks):
        if ma is None or mb is None:
            continue
        resid.append(s - period * (ma + alpha * (mb - ma)))
        used.append((k, ma, mb, alpha))
    if len(resid) < 10:
        return {"desen_tik": len(resid)}
    base = statistics.median(resid)
    err = [abs(r - base) * 1000 for r in resid]
    outliers = [(k, ma, mb, round(alpha, 3), round(e, 1))
                for (k, ma, mb, alpha), e in zip(used, err) if e > 5.0]
    return {
        "desen_tik": len(resid),
        "cikis_zaman_hatasi_ms_p50_p99_max": [round(_pct(err, 50), 3), round(_pct(err, 99), 3), round(max(err), 3)],
        "zaman_hatasi_5ms_ustu": len(outliers),
        "ornek_sapmalar(tik,idx_a,idx_b,alpha,ms)": outliers[:6],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Canli hat v0")
    ap.add_argument("--title", required=True)
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out-fps", type=float, default=60.0)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--pattern", action="store_true")
    ap.add_argument("--foreground", action="store_true", help="kaynak pencereyi basta one al")
    ap.add_argument("--no-snap", action="store_true", help="cikisi kaynak izgarasina hizalama")
    ap.add_argument("--proc", default="baseline",
                    choices=["baseline", "rife", "rife-lite", "rife-flow", "rife-lite-flow", "rife-flow-trt"])
    args = ap.parse_args()
    if args.foreground:
        winutil.dpi_aware()
        winutil.bring_front(winutil.find_window(args.title))
        time.sleep(0.5)
    summary = run(LiveConfig(title=args.title, delay=args.delay, seconds=args.seconds,
                             out_fps=args.out_fps, preview=not args.no_preview, pattern=args.pattern,
                             proc=args.proc, snap=not args.no_snap))
    print("--- ozet")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
