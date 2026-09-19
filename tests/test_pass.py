import pytest
import torch

from upscaler.process import make_processor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")


def test_pass_picks_real_frame_and_fits():
    p = make_processor("pass", 1080, 1920)
    a = torch.zeros((1080, 1920, 4), dtype=torch.uint8, device="cuda")
    b = torch.full((1080, 1920, 4), 200, dtype=torch.uint8, device="cuda")
    y = p(a, b, 0.0)
    assert y.shape == (1, 3, 1080, 1920) and int(y.max()) == 0
    assert int(p(a, b, 0.7).min()) == 200  # harman yok, en yakin kare
    s = torch.full((720, 1280, 4), 100, dtype=torch.uint8, device="cuda")
    y = p(s, s, 0.0)
    assert y.shape == (1, 3, 1080, 1920) and abs(float(y.float().mean()) - 100) < 1
