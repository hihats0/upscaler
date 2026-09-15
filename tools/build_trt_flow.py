"""rife-flow akis agini TensorRT motoruna cevirir ve olcer.

Sabit boyut (varsayilan 1080p kaynagin yarisi: 540x960 -> dolgu 576x960), FP16.
Adimlar: PyTorch -> ONNX (weights/trt/*.onnx) -> TensorRT motoru (*.engine) ->
eager ile hiz ve cikti farki karsilastirmasi.

Kullanim:
    .venv/Scripts/python.exe tools/build_trt_flow.py --h 540 --w 960 --version 4.25
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
import torch.nn as nn  # noqa: E402

from upscaler.models.rife_flow import RifeFlow  # noqa: E402
from upscaler.models.trt_engine import TrtEngine  # noqa: E402

OUT = os.path.join(ROOT, "weights", "trt")


class FlowModule(nn.Module):
    def __init__(self, rf: RifeFlow, ph: int, pw: int) -> None:
        super().__init__()
        self.rf = rf
        self.net = rf.net
        div, grid = rf.rife._grid(ph, pw)
        self.register_buffer("div", div.clone())
        self.register_buffer("grid", grid.clone())

    def forward(self, x0, x1, timestep):
        return self.rf.flow_padded(x0, x1, timestep, self.div, self.grid)


def timed(fn, n=50, warm=8) -> float:
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
    ap.add_argument("--h", type=int, default=540)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--version", default="4.25")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rf = RifeFlow(args.version)
    m = rf.rife.multiple
    ph, pw = (args.h + m - 1) // m * m, (args.w + m - 1) // m * m
    tag = f"rifeflow_{args.version}_{pw}x{ph}_fp16"
    onnx_path = os.path.join(OUT, tag + ".onnx")
    engine_path = os.path.join(OUT, tag + ".engine")
    mod = FlowModule(rf, ph, pw).eval()
    x0 = torch.rand(1, 3, ph, pw, device="cuda", dtype=torch.float16)
    x1 = torch.rand_like(x0)
    ts = torch.full((1, 1, ph, pw), 0.5, device="cuda", dtype=torch.float16)

    if not os.path.exists(onnx_path):
        t0 = time.perf_counter()
        with torch.inference_mode():
            torch.onnx.export(mod, (x0, x1, ts), onnx_path, input_names=["x0", "x1", "timestep"],
                              output_names=["flow", "mask"], opset_version=17, dynamo=False)
        print(f"ONNX: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB, {time.perf_counter() - t0:.1f} sn)")

    if not os.path.exists(engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print("ONNX hata:", parser.get_error(i))
                raise SystemExit(1)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        t0 = time.perf_counter()
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise SystemExit("motor olusturulamadi")
        with open(engine_path, "wb") as f:
            f.write(blob)
        print(f"TensorRT motoru: {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB, "
              f"{time.perf_counter() - t0:.0f} sn)")

    eng = TrtEngine(engine_path)
    div, grid = rf.rife._grid(ph, pw)
    with torch.inference_mode():
        ref_flow, ref_mask = rf.flow_padded(x0, x1, ts, div, grid)
        out = eng(x0=x0, x1=x1, timestep=ts)
        dflow = (out["flow"].float() - ref_flow.float()).abs().mean().item()
        dmask = (out["mask"].float() - ref_mask.float()).abs().mean().item()
        eager = timed(lambda: rf.flow_padded(x0, x1, ts, div, grid))
        fast = timed(lambda: eng(x0=x0, x1=x1, timestep=ts))
    print(f"{tag}: eager {eager:.2f} ms | TensorRT {fast:.2f} ms | hizlanma x{eager / fast:.2f} | "
          f"ort. fark akis {dflow:.4f} px, maske {dmask:.4f}")


if __name__ == "__main__":
    main()
