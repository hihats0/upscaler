"""Cikis tiki icin kare secimi (live.py ve watch.py ortak).

Kaynak zamani s -> tampondan iki kare + alpha. Kurallar (olculdu, raporlar hat11/hat12):
- Kaynak ve cikis hizi tam sayiysa s ortak izgaraya (50/60 -> 1/300 sn) oturtulur; bazi tikler tam
  gercek kareye duser, ara oranlar sabit kalir.
- Ardisik iki karede alpha zaman farkindan degil sira numarasindan (damgalar farkli ofset
  tahminleriyle yazilir).
- Yeni (Hat 1.4): iki kare arasinda uzun bosluk (yayin donmasi, pencere simge durumu, reklam gecisi)
  varsa ara kare uretilmez, eski kare tutulur. Yoksa donmus kare ile yeni kare arasinda saniyelerce
  RIFE harmani olurdu.
"""
from __future__ import annotations

import math

from .ring import GpuFrameRing, Pick


def snap_grid(period: float | None, out_fps: float) -> tuple[float, int] | None:
    """Kaynak ve cikis hizi tam sayiysa ortak izgara: (adim sn, kaynak periyodundaki faz sayisi)."""
    if not period:
        return None
    src = 1.0 / period
    if abs(src - round(src)) > 1e-6 or abs(out_fps - round(out_fps)) > 1e-6:
        return None  # 29.97 gibi hizlarda hizalama yok
    lcm = math.lcm(round(src), round(out_fps))
    return 1.0 / lcm, round(lcm / src)


GAP_PERIODS = 3.0  # iki kare arasi bundan uzunsa ara kare yok (tutma)


def select(ring: GpuFrameRing, s: float, out_fps: float, snap: bool = True) -> tuple[Pick | None, float]:
    """(secim ya da None, hizalanmis s). Secim release() ile birakilmali."""
    period = ring.clock.period
    if period is None:
        return None, s
    grid = snap_grid(period, out_fps) if snap else None
    origin = ring.clock._offset
    if grid:
        step, phases = grid
        s = origin + round((s - origin) / step) * step
    p = ring.pick(s)
    if p is None:
        return None, s
    if p.a.stamp.index != p.b.stamp.index and (
            p.b.stamp.discontinuity or p.b.stamp.t - p.a.stamp.t > GAP_PERIODS * period):
        p.alpha = 0.0
        return p, s
    if grid:
        a = p.alpha * phases
        ai = round((s - origin) / period * phases) / phases - p.a.stamp.index
        if p.b.stamp.index == p.a.stamp.index + 1 and -0.5 / phases <= ai <= 1 + 0.5 / phases:
            p.alpha = min(max(ai, 0.0), 1.0)
        elif abs(a - round(a)) < 0.15:  # sira atlamasi ya da tutma: zaman oranina don
            p.alpha = round(a) / phases
    return p, s


def is_real(alpha: float) -> bool:
    return alpha <= 1e-3 or alpha >= 1 - 1e-3
