"""AI yeniden cizim modelleri (upscaler/models/ai.py) -> TensorRT FP16 motoru, ya da hiz taramasi.

TV modu 50 FPS: kare basi 20 ms. Hattin geri kalani ~1 ms; isinma kisitlamasina pay icin ag hedefi
GPU p95 <= ~12 ms (sessiz sistemde olc: arka planda ffmpeg/egitim varken olcum yaniltir).

Tarama: hazir aday adlari (esr-gen-270, esrx2-540 ...) ya da "ainet:<feat>:<conv>:<frames>" (rastgele agirlik).
Motor girdisi/ciktisi BGR 0..255 FP16, [1, 3*frames, 1080, 1920] -> [1, 3, 1080, 1920] (yuvarlanmis).

    .venv/Scripts/python.exe tools/build_trt_ai.py --bench esr-gen-270 ainet:64:8:1 ainet:48:8:3
    .venv/Scripts/python.exe tools/build_trt_ai.py --name ai_v1      # egitilmis agirlik
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from tools.build_trt_repair import time_engine  # noqa: E402
from tools.build_trt_sr import build  # noqa: E402
from upscaler.models.ai import AiBgr255, AiNet, build as build_cand, engine_path, load_ai  # noqa: E402
from upscaler.models.trt_engine import TrtEngine  # noqa: E402

OUT = os.path.join(ROOT, "weights", "trt")


def export(net: nn.Module, frames: int, onnx_path: str, h: int, w: int) -> AiBgr255:
    wrap = AiBgr255(net, frames).half().cuda().eval()
    x = torch.rand(1, 3 * frames, h, w, device="cuda", dtype=torch.float16) * 255
    with torch.inference_mode():
        torch.onnx.export(wrap, (x,), onnx_path, input_names=["x"], output_names=["y"],
                          opset_version=17, dynamo=False)
    return wrap


def time_multi(eng: TrtEngine, frames: int, h: int, w: int) -> dict:
    if frames == 1:
        return time_engine(eng, h, w)
    # time_engine 3 kanal girdi uretir; cok kareli motor icin girdiyi burada ver
    import time
    x = torch.rand(1, 3 * frames, h, w, device="cuda", dtype=torch.float16) * 255
    t_end = time.perf_counter() + 2.0
    while time.perf_counter() < t_end:
        eng(x=x)
    torch.cuda.synchronize()
    gs = []
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(300):
        e0.record()
        eng(x=x)
        e1.record()
        torch.cuda.synchronize()
        gs.append(e0.elapsed_time(e1))
    gs.sort()
    return {"gpu_p50": round(gs[150], 2), "gpu_p95": round(gs[285], 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*", default=None)
    ap.add_argument("--name", default="", help="weights/ai/<ad>.pth")
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--w", type=int, default=1920)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.bench:
        tmp = os.path.join(OUT, "bench_ai")
        os.makedirs(tmp, exist_ok=True)
        path = os.path.join(ROOT, "runs", "ai_bench.json")
        res = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        for spec in args.bench:
            if spec.startswith("ainet:"):
                _, feat, conv, frames = spec.split(":")
                net = AiNet(int(feat), int(conv), int(frames))
                nn.init.normal_(net.tail.weight, std=0.01)
                frames = int(frames)
            else:
                net, frames = build_cand(spec), 1
            tag = spec.replace(":", "_") + f"_{args.w}x{args.h}"
            onnx_path, eng_path = os.path.join(tmp, tag + ".onnx"), os.path.join(tmp, tag + ".engine")
            try:
                export(net, frames, onnx_path, args.h, args.w)
                if not os.path.exists(eng_path):
                    build(onnx_path, eng_path, opt_level=2)
                t = time_multi(TrtEngine(eng_path), frames, args.h, args.w)
            except Exception as e:  # noqa: BLE001 (cok buyuk aday motor kuramayabilir)
                t = {"hata": str(e)[:200]}
            params = sum(p.numel() for p in net.parameters())
            res[spec] = {"param_k": round(params / 1000, 1), **t}
            print(spec, json.dumps(res[spec]), flush=True)
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1)
            del net
            torch.cuda.empty_cache()
        return
    net = load_ai(args.name).cuda().eval()
    eng_path = engine_path(args.name, args.h, args.w)
    onnx_path = eng_path.replace(".engine", ".onnx")
    wrap = export(net, net.frames, onnx_path, args.h, args.w)
    build(onnx_path, eng_path)
    os.remove(onnx_path)
    eng = TrtEngine(eng_path)
    x = torch.rand(1, 3 * net.frames, args.h, args.w, device="cuda", dtype=torch.float16) * 255
    with torch.inference_mode():
        ref = wrap(x).float()
        out = eng(x=x)["y"].float()
    diff = (out - ref).abs().mean().item()
    print(json.dumps({"motor": eng_path, "ort_fark_255": round(diff, 4), **time_multi(eng, net.frames, args.h, args.w)}))


if __name__ == "__main__":
    main()
