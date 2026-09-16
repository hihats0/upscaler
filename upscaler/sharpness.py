"""Etkili cozunurluk olcumu (F26, 2026-09-16).

Bir 1080p karenin gercekte ne kadar detay tasidigini SADECE sayi olarak olcer (kare saklanmaz).

Iki olcu, ikisi de kadrajin ortasinda (logo, skor tabelasi, alt bant disarida):
- Kucult-buyut kaybi: kare s oraninda kucultulup (INTER_AREA) geri buyutulur (INTER_CUBIC),
  ortalama mutlak fark kayip(s). Icerik gercekte 720p ise s >= 0,67 neredeyse kayipsizdir.
  Icerikten bagimsiz olsun diye kayip(s) / kayip(0,5) orani kullanilir (oran(s)).
- Radyal spektrum: Nyquist'e gore 0,25-0,5 / 0,5-0,75 / 0,75-1 bantlarinin enerji orani.
"""
from __future__ import annotations

import cv2
import numpy as np

SCALES = (0.9, 0.8, 0.7, 0.6, 0.5)
# kadraj orta bolgesi (satir, sutun oranlari)
CROP = (0.22, 0.78, 0.12, 0.88)
_WIN_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def luma(img: np.ndarray, bgr: bool = True) -> np.ndarray:
    a = img[..., :3].astype(np.float32)
    b, g, r = (a[..., 0], a[..., 1], a[..., 2]) if bgr else (a[..., 2], a[..., 1], a[..., 0])
    return 0.0722 * b + 0.7152 * g + 0.2126 * r


def _crop(y: np.ndarray) -> np.ndarray:
    h, w = y.shape
    return y[int(h * CROP[0]):int(h * CROP[1]), int(w * CROP[2]):int(w * CROP[3])]


def _spectrum_setup(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if shape not in _WIN_CACHE:
        h, w = shape
        win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        fy = np.fft.fftfreq(h)[:, None] * 2.0   # 1.0 = Nyquist
        fx = np.fft.rfftfreq(w)[None, :] * 2.0
        rad = np.sqrt(fy ** 2 + fx ** 2)
        _WIN_CACHE[shape] = (win, rad)
    return _WIN_CACHE[shape]


def measure(y: np.ndarray) -> dict[str, float]:
    """y: tam kare luma (float32, 0-255). Donus: sadece sayilar."""
    h, w = y.shape
    c = _crop(y)
    out: dict[str, float] = {"std": float(c.std())}
    loss = {}
    for s in SCALES:
        small = cv2.resize(y, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        back = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        loss[s] = float(np.abs(_crop(back) - c).mean())
    base = max(loss[0.5], 1e-6)
    out["kayip05"] = loss[0.5]
    for s in SCALES:
        out[f"oran{int(s * 100)}"] = loss[s] / base
    win, rad = _spectrum_setup(c.shape)
    p = np.abs(np.fft.rfft2((c - c.mean()) * win)) ** 2
    e = lambda lo, hi: float(p[(rad >= lo) & (rad < hi)].sum())  # noqa: E731
    e1 = max(e(0.25, 0.5), 1e-6)
    out["spk_50_75"] = e(0.5, 0.75) / e1
    out["spk_75_100"] = e(0.75, 1.0) / e1
    return out


def summarize(rows: list[dict[str, float]], min_std: float = 12.0) -> dict[str, float]:
    """Duz kareler (std < min_std) atilir; her olcunun medyani."""
    good = [r for r in rows if r["std"] >= min_std]
    if not good:
        return {"n": 0}
    keys = [k for k in good[0] if k not in ("t",)]
    res: dict[str, float] = {"n": len(good), "atilan": len(rows) - len(good)}
    for k in keys:
        res[k] = round(float(np.median([r[k] for r in good])), 4)
    return res
