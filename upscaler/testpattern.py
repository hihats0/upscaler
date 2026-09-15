"""Canli hat icin test deseni.

Her karenin ust satirina kare numarasi ikili barkod olarak gomulur. Yakalanan
kareden numara geri okunur. Boylece TOD'a ihtiyac duymadan, ayni canli hattan:
dusen kare, zaman cizgisi hatasi ve gecikme olculur.

Desen: ust satirda BITS adet kare blok (beyaz=1, siyah=0), altinda yatay kayan
dikey cubuk ve daire (ara kare kalitesi icin hareketli icerik).
Geometri layout() icinde, numpy (render) ve pygame (pencere) ayni geometriyi cizer.
"""
from __future__ import annotations

import math

import numpy as np

BITS = 20
BLOCK = 24      # blok kenari (px)
PITCH = 32      # blok araligi (px)
MARGIN = 8
BAR_TOP = MARGIN + BLOCK + 16
BAR_HALF = 6
BALL_R = 14
FIELD = (40, 90, 40)   # simetrik oldugu icin RGB = BGR
WHITE = (255, 255, 255)
BAR = (230, 230, 230)


def layout(index: int, width: int, height: int, fps: float):
    """(bit listesi, cubuk x, (top x, top y))"""
    t = index / fps
    bits = [(index >> b) & 1 for b in range(BITS)]
    bx = int((t * width * 0.5) % width)
    cy = BAR_TOP + (height - BAR_TOP) // 2
    r = (height - BAR_TOP) // 3
    px = int(width / 2 + r * math.cos(2 * math.pi * 0.7 * t))
    py = int(cy + r * math.sin(2 * math.pi * 0.7 * t))
    return bits, bx, (px, py)


def render(index: int, width: int = 1280, height: int = 720, fps: float = 50.0) -> np.ndarray:
    """index numarali kareyi BGRA uint8 olarak uretir."""
    bits, bx, (px, py) = layout(index, width, height, fps)
    img = np.zeros((height, width, 4), np.uint8)
    img[..., 3] = 255
    img[BAR_TOP:, :, :3] = FIELD
    for b, on in enumerate(bits):
        if on:
            x = MARGIN + b * PITCH
            img[MARGIN:MARGIN + BLOCK, x:x + BLOCK, :3] = WHITE
    img[BAR_TOP:, max(0, bx - BAR_HALF):bx + BAR_HALF, :3] = BAR
    y0, y1 = max(0, py - BALL_R), min(height, py + BALL_R + 1)
    x0, x1 = max(0, px - BALL_R), min(width, px + BALL_R + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    ball = (yy - py) ** 2 + (xx - px) ** 2 <= BALL_R ** 2
    img[y0:y1, x0:x1, :3][ball] = WHITE
    return img


def decode(frame: np.ndarray) -> int | None:
    """Yakalanan BGRA karenin barkodundan kare numarasini okur. Okunamazsa None."""
    h, w = frame.shape[:2]
    if h < MARGIN + BLOCK or w < MARGIN + BITS * PITCH:
        return None
    y = MARGIN + BLOCK // 2
    value = 0
    for b in range(BITS):
        x = MARGIN + b * PITCH + BLOCK // 2
        v = int(frame[y, x, 1])
        if 60 < v < 190:  # ne siyah ne beyaz: desen degil ya da bozuk
            return None
        if v >= 190:
            value |= 1 << b
    return value
