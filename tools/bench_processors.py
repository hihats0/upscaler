"""Islemci hiz olcumu: 1080p iki kare + alpha -> 4K (uint8, GPU'da).

Her islemci icin ara kare (alpha=0.5) ve gercek kare (alpha=0) suresini olcer
ve 50 -> 60 FPS icin saniyelik GPU butcesini hesaplar (60 cikisin 50'si ara kare).

Kullanim:
    .venv/Scripts/python.exe tools/bench_processors.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from upscaler.process import make_processor  # noqa: E402


def timed(fn, n=40, warm=6) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    torch.backends.cudnn.benchmark = True
    names = sys.argv[1:] or ["baseline", "rife-flow", "rife-lite-flow", "rife"]
    a = torch.randint(0, 256, (1080, 1920, 4), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 256, (1080, 1920, 4), dtype=torch.uint8, device="cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} | 1920x1080 -> 3840x2160")
    for name in names:
        p = make_processor(name)
        mid = timed(lambda: p(a, b, 0.5))
        real = timed(lambda: p(a, b, 0.0))
        budget = 50 * mid + 10 * real  # 60 cikis/sn: 50 ara, 10 gercek kare
        verdict = "SIGAR" if budget < 850 else "SIGMAZ"
        print(f"{p.name:24s} ara kare {mid:6.1f} ms | gercek kare {real:5.1f} ms | "
              f"50->60 butce {budget:6.0f} ms/sn -> {verdict} (esik 850)")
        del p
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
