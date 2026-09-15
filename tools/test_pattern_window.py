"""50 FPS test deseni penceresi (canli hattin kaynagi olarak).

Kullanim:
    .venv/Scripts/python.exe tools/test_pattern_window.py --fps 50 --seconds 30

Pencere cercevesizdir (WGC yakalamasi = istemci alani). Kare n, n/fps anina
zamanlanir. Cikista kac kare cizildigini ve atlandigini yazar. Diske yazmaz.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TITLE = "upscaler-test-pattern"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--x", type=int, default=40)
    ap.add_argument("--y", type=int, default=40)
    args = ap.parse_args()

    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    ctypes.windll.winmm.timeBeginPeriod(1)  # sleep hassasiyeti 1 ms
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{args.x},{args.y}"
    import pygame

    from upscaler import testpattern as tp

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height), pygame.NOFRAME)
    pygame.display.set_caption(TITLE)
    w, h = args.width, args.height

    t0 = time.perf_counter()
    n = drawn = skipped = 0
    try:
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
            now = time.perf_counter() - t0
            if now > args.seconds:
                return
            due = n / args.fps
            if now < due:
                time.sleep(max(0.0, due - now - 0.001))
                continue
            target = int(now * args.fps)
            if target > n:
                skipped += target - n
                n = target
            bits, bx, (px, py) = tp.layout(n, w, h, args.fps)
            screen.fill((0, 0, 0))
            screen.fill(tp.FIELD, (0, tp.BAR_TOP, w, h - tp.BAR_TOP))
            for b, on in enumerate(bits):
                if on:
                    screen.fill(tp.WHITE, (tp.MARGIN + b * tp.PITCH, tp.MARGIN, tp.BLOCK, tp.BLOCK))
            screen.fill(tp.BAR, (max(0, bx - tp.BAR_HALF), tp.BAR_TOP, 2 * tp.BAR_HALF, h - tp.BAR_TOP))
            pygame.draw.circle(screen, tp.WHITE, (px, py), tp.BALL_R)
            pygame.display.flip()
            drawn += 1
            n += 1
    finally:
        print(f"desen: cizilen={drawn} atlanan={skipped}", flush=True)
        pygame.quit()


if __name__ == "__main__":
    main()
