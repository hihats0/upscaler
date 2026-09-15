import os
import time

import pytest
import torch

from upscaler.models.rife import WEIGHTS

need = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(os.path.join(WEIGHTS, "flownet_v4.25.pkl")),
    reason="CUDA ya da RIFE agirligi yok",
)


@need
@pytest.mark.parametrize("version", ["4.25", "4.25.lite"])
def test_identical_frames_give_same_frame(version):
    from upscaler.models.rife import Rife
    r = Rife(version)
    torch.manual_seed(0)
    img = torch.rand(1, 3, 270, 480, device="cuda").half()
    out = r.interpolate(img, img, 0.5)
    assert out.shape == img.shape
    assert (out.float() - img.float()).abs().mean() < 2 / 255


@need
def test_moving_square_lands_in_the_middle():
    """Sola 40 px kayan kare: t=0.5'te 20 px kaymis olmali (dogrusal harman iki hayalet verirdi)."""
    from upscaler.models.rife import Rife
    r = Rife("4.25")
    a = torch.zeros(1, 3, 256, 256, device="cuda").half()
    b = torch.zeros_like(a)
    a[..., 100:140, 100:140] = 1.0
    b[..., 100:140, 140:180] = 1.0
    out = r.interpolate(a, b, 0.5)[0, 1].float()
    cols = out[100:140].mean(0)
    center = float((cols * torch.arange(256, device="cuda")).sum() / cols.sum())
    assert center == pytest.approx(140.0, abs=4.0)
    assert float(out[120, 125]) > 0.7  # kare dolu (hayalet degil)


@need
def test_processor_extremes_use_real_frames_and_shape():
    from upscaler.process import BaselineProcessor, RifeProcessor
    p = RifeProcessor("4.25.lite")
    base = BaselineProcessor()
    a = torch.randint(0, 256, (1020, 1920, 4), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 256, (1020, 1920, 4), dtype=torch.uint8, device="cuda")
    assert torch.equal(p(a, b, 0.0), base(a, b, 0.0))
    assert p(a, b, 0.4).shape == (1, 3, 2160, 3840)


@need
@pytest.mark.parametrize("version,scale", [("4.25", 1.0), ("4.25", 0.5), ("4.25.lite", 1.0), ("4.25.lite", 0.5)])
def test_speed_report_1080p(version, scale):
    """Olcum (assert degil): 1080p ara kare hizi."""
    from upscaler.models.rife import Rife
    r = Rife(version, scale=scale)
    a = torch.rand(1, 3, 1080, 1920, device="cuda").half()
    b = torch.rand_like(a)
    for _ in range(5):
        r.interpolate(a, b, 0.5)
    torch.cuda.synchronize()
    n = 40
    t0 = time.perf_counter()
    for _ in range(n):
        r.interpolate(a, b, 0.5)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"\nRIFE {version} scale={scale} 1080p FP16 (PyTorch, TensorRT yok): {ms:.1f} ms = {1000 / ms:.0f} kare/sn")
