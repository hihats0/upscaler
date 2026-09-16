"""Canli puan yardimcilari (GPU gerekir: YMetric)."""
import torch

from upscaler.score import GtBank, LiveScorer, YMetric, luma_bgr


def fake_bank(shift=0):
    b = GtBank.__new__(GtBank)
    b.meta = {"gt": "x", "gt_start_s": 0, "gt_fps": 60, "sim_fps": 50}
    b.shift, b.keep, b.period, b.ratio = shift, (0, 3), 6, 1.2
    b.frames = {g: None for g in range(0, 600) if b.wanted(g)}
    b.load_s = 0
    return b


def test_content_pos_and_gt_index():
    cp = LiveScorer.content_pos
    assert cp(10, 11, 0.0) == 10
    assert cp(10, 11, 1.0) == 11
    assert abs(cp(12, 13, 0.5) - 12.5) < 1e-9
    assert cp(12, 14, 0.25) == 12.5          # dusen kare: iki kare arasi
    assert cp(499, 0, 0.5) is None           # dongu basi
    assert cp(None, 3, 0.0) is None
    b = fake_bank()
    assert b.gt_index(10.0) == 12            # gercek kare tiki: i % 5 == 0
    assert b.gt_index(12.5) == 15            # alpha 0,5 ara kare -> g % 6 == 3
    assert b.gt_index(11.0) is None          # g = 13,2 tam sayi degil
    b1 = fake_bank(shift=1)
    assert b1.gt_index(10.0) == 13


def test_metric_identity_and_shift():
    m = YMetric()
    torch.manual_seed(0)
    a = torch.rand(256, 256, device="cuda") * 255
    p, s = m(a, a.clone())
    assert p > 90 and abs(s - 1) < 1e-4
    p2, s2 = m(a, torch.roll(a, 3, dims=1))
    assert p2 < 15 and s2 < 0.2


def test_luma_bgr():
    y = torch.zeros(1, 3, 2, 2, dtype=torch.uint8, device="cuda")
    y[0, 1] = 255
    assert abs(float(luma_bgr(y)[0, 0]) - 0.7152 * 255) < 1e-3


def test_async_offer_scores_and_matches_sync():
    b = fake_bank()
    torch.manual_seed(1)
    gt = (torch.rand(2160, 3840, device="cuda") * 255).to(torch.uint8)
    from upscaler.score import CROP
    b.frames = {12: gt[CROP].cpu().numpy(), 15: gt[CROP].cpu().numpy()}
    sc = LiveScorer(b)
    y = gt.expand(1, 3, 2160, 3840).clone()  # gri: Y = deger
    sc.offer(y, 10, 11, 0.0, t=1.0, budget_s=0.01)
    sc.finish()
    assert len(sc.rows) == 1, sc.summary()
    t, g, cls, alpha, psnr, ssim = sc.rows[0]
    assert g == 12 and cls == "gercek" and psnr > 45 and ssim > 0.99
    sc.offer(y, 12, 13, 0.5, t=1.01, budget_s=0.01)  # oran siniri (g=15)
    assert sc.skipped_rate == 1


def test_cpu_metric_matches_gpu():
    import numpy as np
    from upscaler.score import ssim_psnr_cpu
    torch.manual_seed(2)
    a = torch.rand(300, 400, device="cuda") * 255
    b = (a + torch.randn_like(a) * 8).clamp(0, 255)
    pg, sg = YMetric()(a, b)
    pc, sc_ = ssim_psnr_cpu(a.cpu().numpy().astype(np.float32), b.cpu().numpy().astype(np.float32))
    assert abs(pg - pc) < 0.01 and abs(sg - sc_) < 2e-3
