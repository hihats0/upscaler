"""RIFE hiz olcumu: normal (eager) vs CUDA graphs.

Hafif modellerde darbogaz cogu zaman GPU hesabi degil, Python'dan gelen yuzlerce
kucuk cekirdek cagrisidir. CUDA graphs sabit boyutlu cagriyi bir kez kaydedip
tek seferde yeniden oynatir.

Kullanim:
    .venv/Scripts/python.exe tools/bench_rife.py --h 1080 --w 1920
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from upscaler.models.rife import Rife  # noqa: E402


def timed(fn, n=40, warm=5) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--w", type=int, default=1920)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True

    print(f"GPU: {torch.cuda.get_device_name(0)} | {args.w}x{args.h} FP16")
    for version, scale in (("4.25", 1.0), ("4.25", 0.5), ("4.25.lite", 1.0), ("4.25.lite", 0.5)):
        r = Rife(version, scale=scale)
        a = torch.rand(1, 3, args.h, args.w, device="cuda").half()
        b = torch.rand_like(a)
        eager = timed(lambda: r.interpolate(a, b, 0.5))
        graphed = r.cuda_graph(args.h, args.w)
        g = timed(lambda: graphed(a, b, 0.5))
        # dogruluk: graph ciktisi eager ile ayni mi?
        diff = (graphed(a, b, 0.3).float() - r.interpolate(a, b, 0.3).float()).abs().max().item()
        print(f"RIFE {version:9s} scale={scale}: eager {eager:6.1f} ms ({1000 / eager:5.0f} fps) | "
              f"CUDA graph {g:6.1f} ms ({1000 / g:5.0f} fps) | max fark {diff:.2e}")
        del r, graphed
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
