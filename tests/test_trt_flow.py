import os

import pytest
import torch

from upscaler.models.rife import ROOT

ENGINE = os.path.join(ROOT, "weights", "trt", "rifeflow_4.25_960x576_fp16.engine")
need = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.exists(ENGINE),
                          reason="CUDA ya da TensorRT motoru yok (tools/build_trt_flow.py)")


@need
def test_trt_processor_matches_eager_on_pattern():
    """TensorRT akisiyla uretilen 4K ara kare, PyTorch akisiyla uretilenle neredeyse ayni olmali."""
    from upscaler.process import RifeFlowProcessor
    from upscaler.testpattern import render
    eager = RifeFlowProcessor("4.25")
    fast = RifeFlowProcessor("4.25", trt=True)
    a = torch.from_numpy(render(200, 1920, 1080)).cuda()
    b = torch.from_numpy(render(201, 1920, 1080)).cuda()
    for alpha in (0.25, 0.5, 0.833):
        ye = eager(a, b, alpha).float()
        yt = fast(a, b, alpha).float()
        assert yt.shape == (1, 3, 2160, 3840)
        assert (ye - yt).abs().mean() < 0.5
    assert fast._engines, "motor yuklenmedi"
    assert next(iter(fast._engines.values())) is not None
