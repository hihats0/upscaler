"""SR modelleri: agirlik yukleme, reparametrizasyon denkligi, FP16, boyut."""
import os

import pytest
import torch
import torch.nn.functional as F

from upscaler.models.rife import ROOT

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")
has_rt4ksr = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "weights/rt4ksr/rt4ksr_x2.pth")),
                                reason="rt4ksr agirligi yok")


def _smooth_image(h, w, device="cuda"):
    """Dogal goruntuye benzer yumusak desen (rastgele gurultu SR icin anlamsiz)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    small = torch.rand(1, 3, h // 16, w // 16, generator=g)
    return F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1).to(device)


@cuda
@has_rt4ksr
def test_rt4ksr_rep_matches_train_form():
    from upscaler.models.sr import load_sr
    train = load_sr("rt4ksr-x2", rep=False).float()
    rep = load_sr("rt4ksr-x2", rep=True).float()
    x = _smooth_image(128, 200)
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False  # TF32 konvolusyonu ~1e-3 fark uretir, denkligi gizler
    try:
        with torch.inference_mode():
            a, b = train(x), rep(x)
    finally:
        torch.backends.cudnn.allow_tf32 = tf32
    assert a.shape == (1, 3, 256, 400)
    assert (a - b).abs().max().item() < 1e-4


@cuda
@has_rt4ksr
def test_rt4ksr_fp16_close_and_flat_region_finite():
    from upscaler.models.sr import load_sr
    net = load_sr("rt4ksr-x2")
    x = _smooth_image(128, 200)
    x[:, :, :40] = 0.0  # siyah bant (TOD'da letterbox): LayerNorm 0/0 olmamali
    with torch.inference_mode():
        ref = net.float()(x)
        half = net.half()(x.half()).float()
    assert torch.isfinite(half).all()
    assert (ref - half).abs().mean().item() * 255 < 0.2


@cuda
@has_rt4ksr
def test_rt4ksr_output_is_sane_upscale():
    """Cikti bicubic buyutmeye yakin olmali (renk/konum kaymasi, cop cikti yok).

    Kalite kiyasi dogal goruntuyle yapildi (rapor: RT4KSR 35,6 dB, bicubic 32,7 dB); sentetik
    desenler (keskin ikili kenar) dogal goruntu istatistigi tasimadigi icin kalite testi olamaz.
    """
    from upscaler.models.sr import load_sr
    net = load_sr("rt4ksr-x2").float()
    hr = _smooth_image(256, 384)
    lr = F.interpolate(hr, scale_factor=0.5, mode="bicubic", antialias=True, align_corners=False)
    with torch.inference_mode():
        sr = net(lr).clamp(0, 1)
    bic = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
    c = (slice(None), slice(None), slice(8, -8), slice(8, -8))
    psnr = 10 * torch.log10(1 / F.mse_loss(sr[c], bic[c]))
    assert psnr.item() > 30


def _bgra(h, w, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    small = torch.rand(1, 3, h // 8, w // 8, generator=g)
    img = F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
    bgra = torch.cat([img, torch.ones(1, 1, h, w)], 1)[0].permute(1, 2, 0)
    return (bgra * 255).round().to(torch.uint8).cuda()


@cuda
@has_rt4ksr
def test_split_left_sr_right_bicubic_with_divider():
    from upscaler.process import SrProcessor
    a = _bgra(64, 96, 1)
    kw = dict(sr_name="rt4ksr-x2", interp="lerp", out_h=128, out_w=192, trt=False)
    sr_only = SrProcessor(split=False, **kw)(a, a, 0.0).clone()
    split = SrProcessor(split=True, **kw)(a, a, 0.0).clone()
    half = 96
    assert torch.equal(split[..., :half - 2], sr_only[..., :half - 2])
    mid = a[..., :3].permute(2, 0, 1).unsqueeze(0).half() / 255.0  # BGR 0..1
    bic = F.interpolate(mid * 255.0, size=(128, 192), mode="bicubic", align_corners=False)
    bic = bic.add_(0.5).clamp_(0, 255).to(torch.uint8)
    assert torch.equal(split[..., half + 2:], bic[..., half + 2:])
    assert (split[..., half - 2:half + 2] == 255).all()
    assert not torch.equal(split[..., half + 2:], sr_only[..., half + 2:])  # sag yari gercekten farkli


@cuda
@has_rt4ksr
def test_sr_letterbox_keeps_aspect_and_black_bars():
    from upscaler.process import SrProcessor
    a = _bgra(60, 96, 2)  # 16:10 -> 192x120 sigar, ust/alt 4 px bant
    out = SrProcessor("rt4ksr-x2", "lerp", False, out_h=128, out_w=192, trt=False)(a, a, 0.0)
    assert out.shape == (1, 3, 128, 192)
    assert (out[:, :, :4] == 0).all() and (out[:, :, 124:] == 0).all()
    assert out[:, :, 4:124].float().mean() > 20
