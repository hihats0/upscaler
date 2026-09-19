"""Onarim agi (upscaler/models/repair.py) -> TensorRT motoru, ya da boyut taramasi.

Tarama (B0): rastgele agirlikli farkli genislik/derinlikleri 1080p FP16 motora cevirip olcer.
TOD 50 FPS: kare basi 20 ms. Hattin geri kalani (secim, sunum, bilgi) ~1 ms; guvenlik payi ve
isinma kisitlamasi icin ag hedefi p95 <= 10 ms.

    .venv/Scripts/python.exe tools/build_trt_repair.py --bench 16:2 32:4 48:6
    .venv/Scripts/python.exe tools/build_trt_repair.py --name rep_v0      # egitilmis agirlik
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from tools.build_trt_sr import build  # noqa: E402
from upscaler.models.repair import RepairBgr255, RepairNet, engine_path, load_repair  # noqa: E402
from upscaler.models.trt_engine import TrtEngine  # noqa: E402

OUT = os.path.join(ROOT, "weights", "trt")


def export(net: RepairNet, onnx_path: str, h: int, w: int) -> None:
    wrap = RepairBgr255(net).half().cuda().eval()
    x = torch.rand(1, 3, h, w, device="cuda", dtype=torch.float16) * 255
    with torch.inference_mode():
        torch.onnx.export(wrap, (x,), onnx_path, input_names=["x"], output_names=["y"],
                          opset_version=17, dynamo=False)


def time_engine(eng: TrtEngine, h: int, w: int, n: int = 300) -> dict:
    """Duvar saati (CPU dahil, canli hattaki gibi) ve CUDA olayi (sadece GPU) sureleri, ms.
    Once ~2 sn isinma: GPU saat hizi dusukken olcum yaniltir."""
    x = torch.rand(1, 3, h, w, device="cuda", dtype=torch.float16) * 255
    t_end = time.perf_counter() + 2.0
    while time.perf_counter() < t_end:
        eng(x=x)
    torch.cuda.synchronize()
    ts, gs = [], []
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(n):
        t0 = time.perf_counter()
        e0.record()
        eng(x=x)
        e1.record()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
        gs.append(e0.elapsed_time(e1))
    ts.sort()
    gs.sort()
    return {"p50": round(ts[n // 2], 2), "p95": round(ts[int(n * 0.95)], 2), "max": round(ts[-1], 2),
            "gpu_p50": round(gs[n // 2], 2), "gpu_p95": round(gs[int(n * 0.95)], 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*", default=None, help="ch:blocks listesi")
    ap.add_argument("--name", default="", help="weights/repair/<ad>.pth")
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--w", type=int, default=1920)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.bench:
        tmp = os.path.join(OUT, "bench_repair")
        os.makedirs(tmp, exist_ok=True)
        res = []
        for spec in args.bench:
            ch, bl = map(int, spec.split(":"))
            net = RepairNet(ch, bl)
            torch.nn.init.normal_(net.tail.weight, std=0.01)  # sifir agirlik: gercekci olmayan olcum olmasin
            tag = f"rep_c{ch}_b{bl}_{args.w}x{args.h}"
            onnx_path, eng_path = os.path.join(tmp, tag + ".onnx"), os.path.join(tmp, tag + ".engine")
            export(net, onnx_path, args.h, args.w)
            if not os.path.exists(eng_path):
                build(onnx_path, eng_path, opt_level=2)  # tarama: hizli kurulum, sadece boyut karari
            t = time_engine(TrtEngine(eng_path), args.h, args.w)
            params = sum(p.numel() for p in net.parameters())
            row = {"ch": ch, "blocks": bl, "param_k": round(params / 1000, 1), **t}
            print(json.dumps(row), flush=True)
            res.append(row)
            os.remove(onnx_path)
        with open(os.path.join(ROOT, "runs", "repair_bench.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        return
    net = load_repair(args.name)
    eng_path = engine_path(args.name, args.h, args.w)
    onnx_path = eng_path.replace(".engine", ".onnx")
    export(net, onnx_path, args.h, args.w)
    build(onnx_path, eng_path)
    eng = TrtEngine(eng_path)
    x = torch.rand(1, 3, args.h, args.w, device="cuda", dtype=torch.float16) * 255
    with torch.inference_mode():
        ref = RepairBgr255(net).half()(x).float()
        out = eng(x=x)["y"].float()
    diff = (out - ref).abs().mean().item()
    print(json.dumps({"motor": eng_path, "ort_fark_255": round(diff, 4), **time_engine(eng, args.h, args.w)}))


if __name__ == "__main__":
    main()
