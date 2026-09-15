import pytest
import torch
import torch.nn.functional as F

from upscaler.process import BaselineProcessor, fit_size

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")


def test_fit_size_16_9_and_browser_window():
    assert fit_size(1080, 1920, 2160, 3840) == (2160, 3840, 0, 0)
    assert fit_size(720, 1280, 2160, 3840) == (2160, 3840, 0, 0)
    assert fit_size(1020, 1920, 2160, 3840) == (2040, 3840, 60, 0)


@cuda
def test_output_shape_and_dtype():
    a = torch.randint(0, 256, (1080, 1920, 4), dtype=torch.uint8, device="cuda")
    y = BaselineProcessor()(a, a, 0.0)
    assert y.shape == (1, 3, 2160, 3840) and y.dtype == torch.uint8


@cuda
def test_alpha_zero_is_bicubic_of_a_and_alpha_one_is_b():
    a = torch.randint(0, 256, (90, 160, 4), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 256, (90, 160, 4), dtype=torch.uint8, device="cuda")
    p = BaselineProcessor(180, 320, dtype=torch.float32)
    ref = F.interpolate(a[..., :3].permute(2, 0, 1)[None].float(), size=(180, 320),
                        mode="bicubic", align_corners=False).clamp(0, 255).round().to(torch.uint8)
    assert torch.equal(p(a, b, 0.0), ref)
    refb = F.interpolate(b[..., :3].permute(2, 0, 1)[None].float(), size=(180, 320),
                         mode="bicubic", align_corners=False).clamp(0, 255).round().to(torch.uint8)
    assert torch.equal(p(a, b, 1.0), refb)


@cuda
def test_letterbox_pads_black():
    a = torch.full((1020, 1920, 4), 200, dtype=torch.uint8, device="cuda")
    y = BaselineProcessor()(a, a, 0.0)
    assert int(y[..., :60, :].max()) == 0 and int(y[..., 2100:, :].max()) == 0
    assert int(y[..., 1000, 1920].min()) == 200
