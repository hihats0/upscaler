"""SR modelini TensorRT motoruna cevirir ve olcer (sabit boyut, FP16).

Adimlar: PyTorch -> ONNX (weights/trt/*.onnx) -> TensorRT motoru (*.engine) ->
eager ile hiz ve cikti farki karsilastirmasi. Girdi/cikti RGB 0..1, [1,3,H,W].

Kullanim:
    .venv/Scripts/python.exe tools/build_trt_sr.py --model efrlfn-x2 --h 1080 --w 1920
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tensorrt as trt  # noqa: E402
import torch  # noqa: E402

from upscaler.models.sr import load_sr  # noqa: E402
from upscaler.models.trt_engine import TrtEngine  # noqa: E402

OUT = os.path.join(ROOT, "weights", "trt")


def timed(fn, n=30, warm=8) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def build(onnx_path: str, engine_path: str) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("ONNX hata:", parser.get_error(i))
            raise SystemExit(1)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.builder_optimization_level = 5
    t0 = time.perf_counter()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise SystemExit("motor olusturulamadi")
    with open(engine_path, "wb") as f:
        f.write(blob)
    print(f"TensorRT motoru: {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB, "
          f"{time.perf_counter() - t0:.0f} sn)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efrlfn-x2")
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    net = load_sr(args.model).half()
    tag = f"sr_{args.model}_{args.w}x{args.h}_fp16"
    onnx_path = os.path.join(OUT, tag + ".onnx")
    engine_path = os.path.join(OUT, tag + ".engine")
    x = torch.rand(1, 3, args.h, args.w, device="cuda", dtype=torch.float16)

    if args.force or not os.path.exists(onnx_path):
        with torch.inference_mode():
            torch.onnx.export(net, (x,), onnx_path, input_names=["x"], output_names=["y"],
                              opset_version=17, dynamo=False)
        print(f"ONNX: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.2f} MB)")
    if args.force or not os.path.exists(engine_path):
        build(onnx_path, engine_path)

    eng = TrtEngine(engine_path)
    with torch.inference_mode():
        ref = net(x).float()
        out = eng(x=x)["y"].float()
        diff = (out - ref).abs().mean().item() * 255
        eager = timed(lambda: net(x))
        fast = timed(lambda: eng(x=x))
    print(f"{tag}: eager {eager:.2f} ms | TensorRT {fast:.2f} ms | x{eager / fast:.1f} | "
          f"ort. fark {diff:.4f}/255 | cikti {tuple(out.shape)}")


if __name__ == "__main__":
    main()
