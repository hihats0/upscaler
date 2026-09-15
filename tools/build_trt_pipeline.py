"""Kaynasik canli hat motorlarini (models/fused.py) derler, dogrular ve olcer.

Varsayilan iki motor: still (gercek kare: BGRA -> SR -> 4K uint8) ve rgbsr (ara kare karma yol:
RGB -> SR -> 4K uint8). --full: tam kaynasik ara kare motoru (akis + warp + SR) da derlenir ve olculur.
Dogrulama: ayni girdide ayri parcali SrProcessor (TensorRT) ile cikti farki.

Kullanim:
    .venv/Scripts/python.exe tools/build_trt_pipeline.py --h 1020 --w 1920
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from build_trt_sr import build  # noqa: E402
from upscaler.models.fused import InterpSR, RgbSR, StillSR, engine_paths  # noqa: E402
from upscaler.models.sr import load_sr, sr_scale  # noqa: E402
from upscaler.models.trt_engine import TrtEngine  # noqa: E402
from upscaler.process import SrProcessor  # noqa: E402


def timed(fn, n=40, warm=8) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def test_frames(h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Dogal goruntuye benzeyen yumusak desen ve 3-6 px kaydirilmis hali (hareket)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    small = torch.rand(1, 3, h // 12, w // 12, generator=g)
    img = F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
    img = torch.cat([img, torch.ones(1, 1, h, w)], 1)[0].permute(1, 2, 0)
    a = (img * 255).round().to(torch.uint8).cuda().contiguous()
    b = torch.roll(a, shifts=(3, 6), dims=(0, 1)).contiguous()
    return a, b


def ensure(module, inputs, names, path, force) -> TrtEngine:
    onnx_path = path.replace(".engine", ".onnx")
    if force or not os.path.exists(onnx_path):
        with torch.inference_mode():
            torch.onnx.export(module, inputs, onnx_path, input_names=names, output_names=["y"],
                              opset_version=17, dynamo=False)
        print(f"ONNX: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.2f} MB)")
    if force or not os.path.exists(path):
        build(onnx_path, path)
    return TrtEngine(path)


def report(name, t_ref, t_fast, got, want) -> None:
    d = (got.float() - want.float()).abs()
    print(f"{name}: ayri parcalar {t_ref:.2f} ms | motor {t_fast:.2f} ms | fark ort {d.mean():.3f} "
          f"p99.9 {torch.quantile(d.flatten()[::97], 0.999):.1f} /255 | cikti {tuple(got.shape)} {got.dtype}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=1020)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--sr", default="rt4ksr-x2")
    ap.add_argument("--out-h", type=int, default=2160)
    ap.add_argument("--out-w", type=int, default=3840)
    ap.add_argument("--full", action="store_true", help="tam kaynasik ara kare motorunu da derle/olc")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    H, W, oh, ow = args.h, args.w, args.out_h, args.out_w

    sr = load_sr(args.sr).half()
    scale = sr_scale(args.sr)
    paths = engine_paths(args.sr, H, W, oh, ow)
    a, b = test_frames(H, W)
    x = torch.rand(1, 3, H, W, dtype=torch.float16, device="cuda")
    ref = SrProcessor(args.sr, "rife-flow", False, oh, ow, trt=True)

    e_still = ensure(StillSR(sr, H, W, oh, ow, scale).eval(), (a,), ["a"], paths["still"], args.force)
    e_rgbsr = ensure(RgbSR(sr, H, W, oh, ow, scale).eval(), (x,), ["x"], paths["rgbsr"], args.force)

    with torch.inference_mode():
        want = ref(a, b, 0.0).clone()
        report("gercek kare (still)", timed(lambda: ref(a, b, 0.0)), timed(lambda: e_still(a=a)),
               e_still(a=a)["y"], want)
        want = ref(a, b, 0.5).clone()
        flow = ref.flow
        got = e_rgbsr(x=flow.mid_rgb(a, b, 0.5))["y"]
        report("ara kare (karma: akis+warp ayri, SR+cikti motor)", timed(lambda: ref(a, b, 0.5)),
               timed(lambda: e_rgbsr(x=flow.mid_rgb(a, b, 0.5))), got, want)

    if args.full:
        from upscaler.models.rife_flow import RifeFlow
        t = torch.full((1, 1, 1, 1), 0.5, dtype=torch.float16, device="cuda")
        interp = InterpSR(RifeFlow("4.25"), sr, H, W, oh, ow, scale).eval().cuda()
        e_interp = ensure(interp, (a, b, t), ["a", "b", "t"], paths["interp"], args.force)
        with torch.inference_mode():
            report("ara kare (tam kaynasik)", timed(lambda: ref(a, b, 0.5)),
                   timed(lambda: e_interp(a=a, b=b, t=t)), e_interp(a=a, b=b, t=t)["y"], want)


if __name__ == "__main__":
    main()
