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

from . import schedule, winutil
from .capture import WgcSource
from .process import PROCESSORS, make_processor
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
    proc: str = "baseline"  # process.PROCESSORS
    # CSV yolu: benzersiz kare zamanlari + <ad>_ticks.csv cikis tikleri (sadece sayi, goruntu yok)
    dump_timing: str | None = None
    snap: bool = True  # cikisi kaynak+cikis ortak izgarasina hizala
    sr: str = "rt4ksr-x2"  # *-sr islemcilerinin SR modeli
    split: bool = False  # kiyas: sol yari SR, sag yari bicubic


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
        # Seyreltme (onizleme kalitesi onemsiz): 4K float alan kucultme + kopya islem suresine
        # ~3-4 ms ekliyordu (TOD, 2026-09-15).
        sy, sx = max(bgr.shape[2] // self.h, 1), max(bgr.shape[3] // self.w, 1)
        small = bgr[0, :, ::sy, ::sx][:, :self.h, :self.w]
        rgb_xy = small.flip(0).permute(2, 1, 0).contiguous().cpu().numpy()
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
    proc = make_processor(cfg.proc, cfg.out_h, cfg.out_w, sr=cfg.sr, split=cfg.split)
    # Isinma: ilk CUDA cagrilari (cekirdek yukleme, bellek) dongude gec tik yaratmasin.
    for h, w in ((1080, 1920), (1020, 1920), (720, 1280)):
        z = torch.zeros((h, w, 4), dtype=torch.uint8, device="cuda")
        for _ in range(3):
            proc(z, z, 0.5)
    torch.cuda.synchronize()
    decode = None
    if cfg.pattern:
        from .testpattern import decode

    timing: list[tuple] = []  # (raw, barkod, sira, ideal t, tekrar): sadece sayi, tekrar oynatma icin
    tick_log: list[tuple] = []

    def on_new_frame(raw: float, bgra: np.ndarray) -> None:
        meta = decode(bgra) if decode else None
        stamp = ring.push(raw, bgra, meta)
        if cfg.dump_timing:
            timing.append((raw, -1 if meta is None else meta, stamp.index, stamp.t,
                           int(bool(getattr(stamp, "repeat", False)))))

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
    pending_late = 0

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
                pending_late += missed
                k += missed
                continue

            # Saat kilitlenmeden cikis yok; izgara hizalama ve alpha kurallari schedule.select icinde.
            p, s = schedule.select(ring, T - cfg.delay, cfg.out_fps, cfg.snap)
            if p is None:
                waiting += 1
                k += 1
                continue
            real_ticks += schedule.is_real(p.alpha)
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
            if cfg.dump_timing:
                tick_log.append((k, T - t0, (t_a - T) * 1000, s, p.alpha, p.a.stamp.index, p.b.stamp.index,
                                 p.a.stamp.t, p.b.stamp.t, int(p.hold), proc_ms[-1], pending_late, len(ring)))
                pending_late = 0
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
            f.write("raw_s,barcode,index,t,repeat\n")
            f.writelines(f"{r:.7f},{c},{i},{t:.7f},{rep}\n" for r, c, i, t, rep in list(timing))
        stem, ext = os.path.splitext(cfg.dump_timing)
        with open(f"{stem}_ticks{ext or '.csv'}", "w", encoding="utf-8") as f:
            f.write("k,T_s,start_late_ms,s,alpha,a_index,b_index,a_t,b_t,hold,proc_ms,late_before,ring_len\n")
            for (kk, Ts, sl, ss, al, ai, bi, at, bt, hd, pm, lb, rl) in tick_log:
                f.write(f"{kk},{Ts:.6f},{sl:.3f},{ss:.7f},{al:.5f},{ai},{bi},{at:.7f},{bt:.7f},{hd},{pm:.3f},{lb},{rl}\n")
    summary = {
        "sure_sn": round(elapsed, 2),
        "kaynak": f"{ring.shape[1]}x{ring.shape[0]}" if ring.shape else None,
        "giris_fps": round(in_fps, 2) if in_fps else None,
        "wgc_teslim_fps": round(src.delivered / elapsed, 2),
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
    ap.add_argument("--proc", default="baseline", choices=PROCESSORS)
    ap.add_argument("--sr", default="rt4ksr-x2", help="*-sr islemcilerinin SR modeli")
    ap.add_argument("--split", action="store_true", help="kiyas: sol yari SR, sag yari bicubic")
    ap.add_argument("--dump-timing", metavar="CSV",
                    help="kare zamanlarini ve cikis tiklerini CSV'ye yaz (goruntu yazilmaz)")
    args = ap.parse_args()
    if args.foreground:
        winutil.dpi_aware()
        winutil.bring_front(winutil.find_window(args.title))
        time.sleep(0.5)
    summary = run(LiveConfig(title=args.title, delay=args.delay, seconds=args.seconds,
                             out_fps=args.out_fps, preview=not args.no_preview, pattern=args.pattern,
                             proc=args.proc, snap=not args.no_snap, sr=args.sr, split=args.split,
                             dump_timing=args.dump_timing))
    print("--- ozet")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
