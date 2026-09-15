import os

import pytest
import torch

from upscaler.models.rife import WEIGHTS

need = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(os.path.join(WEIGHTS, "flownet_v4.25.pkl")),
    reason="CUDA ya da RIFE agirligi yok",
)


def square_pair(size=512, side=80, shift=80):
    a = torch.zeros(1, 3, size, size, device="cuda").half()
    b = torch.zeros_like(a)
    y0, x0 = size // 2 - side // 2, size // 2 - shift // 2 - side // 2
    a[..., y0:y0 + side, x0:x0 + side] = 1.0
    b[..., y0:y0 + side, x0 + shift:x0 + shift + side] = 1.0
    return a, b, y0, x0


def column_center(img, y0, side):
    cols = img[0, 1, y0:y0 + side].float().mean(0)
    idx = torch.arange(img.shape[3], device=img.device)
    return float((cols * idx).sum() / cols.sum())


@need
def test_flow_at_half_res_then_warp_full_res_lands_in_the_middle():
    """512 px'te 80 px kayan kare. Akis 256 px'te hesaplanir, kaydirma 512 px'te."""
    from upscaler.models.rife_flow import RifeFlow
    rf = RifeFlow("4.25")
    a, b, y0, x0 = square_pair()
    sa = torch.nn.functional.interpolate(a, scale_factor=0.5, mode="bilinear", antialias=True)
    sb = torch.nn.functional.interpolate(b, scale_factor=0.5, mode="bilinear", antialias=True)
    for t in (0.25, 0.5, 0.75):
        flow, mask = rf.flow(sa, sb, t)
        out = rf.synthesize(a, b, flow, mask)
        expected = x0 + 40 + 80 * t  # kare merkezi
        assert column_center(out, y0, 80) == pytest.approx(expected, abs=6.0), t
        assert float(out[0, 1, y0 + 40, round(expected)]) > 0.7  # dolu, hayalet degil


@need
def test_identical_frames_stay_identical():
    from upscaler.models.rife_flow import RifeFlow
    rf = RifeFlow("4.25")
    torch.manual_seed(1)
    img = torch.rand(1, 3, 540, 960, device="cuda").half()
    small = torch.nn.functional.interpolate(img, scale_factor=0.5, mode="bilinear", antialias=True)
    flow, mask = rf.flow(small, small, 0.5)
    out = rf.synthesize(img, img, flow, mask)
    assert float(flow.abs().mean()) < 0.5
    assert (out.float() - img.float()).abs().mean() < 3 / 255


@need
def test_processor_shape_and_extremes():
    from upscaler.process import BaselineProcessor, RifeFlowProcessor
    p = RifeFlowProcessor("4.25")
    base = BaselineProcessor()
    a = torch.randint(0, 256, (1080, 1920, 4), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 256, (1080, 1920, 4), dtype=torch.uint8, device="cuda")
    assert torch.equal(p(a, b, 0.0), base(a, b, 0.0))
    assert p(a, b, 0.5).shape == (1, 3, 2160, 3840)
