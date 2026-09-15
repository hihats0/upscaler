"""Hat 1.4 paket 1: kaynak boyutu degisince hat cokmeden yeniden yapilandirilir.

- plan_input: pencere boyutu icin motor tuvali secimi (tam eslesme, sigdirma, yok).
- fit_bgra: en-boy korunarak tuvale sigdirma (siyah bant, opak alfa).
- GpuFrameRing: islemdeki bir secim varken boyut degisirse eski kareler gecerli kalir,
  slot sayimi bozulmaz.
- FusedSrProcessor: motoru olmayan boyut sigdirma yoluna, hic tuval yoksa ayri parcalara duser.
"""
import os

import numpy as np
import pytest
import torch

from upscaler.process import fit_bgra, plan_input

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")
CANVASES = [(1080, 1920), (1020, 1920)]


def test_plan_exact_match_wins():
    assert plan_input(1080, 1920, CANVASES) == (1080, 1920)
    assert plan_input(1020, 1920, CANVASES) == (1020, 1920)


def test_plan_other_sizes_fit_into_largest_content():
    # 720p (ABR ya da kucuk pencere): 1920x1080 tuvaline tam oturur
    assert plan_input(720, 1280, CANVASES) == (1080, 1920)
    # Normal pencere (baslik cubugu + kenar): en buyuk icerik alanini veren tuval
    assert plan_input(700, 1200, CANVASES) == (1080, 1920)
    # Tuvalden buyuk kaynak da sigdirilir (kucultme)
    assert plan_input(1440, 2560, CANVASES) == (1080, 1920)


def test_plan_without_canvases_is_none():
    assert plan_input(1080, 1920, []) is None


@cuda
def test_fit_bgra_keeps_aspect_with_black_bars():
    src = torch.full((60, 80, 4), 200, dtype=torch.uint8, device="cuda")  # 4:3
    out = fit_bgra(src, 90, 160)
    assert out.shape == (90, 160, 4)
    # 4:3 -> 120x90, sol/sag 20 px siyah
    assert (out[:, :20, :3] == 0).all() and (out[:, 140:, :3] == 0).all()
    assert (out[:, 21:139, :3] == 200).all()
    assert (out[..., 3] == 255).all()


@cuda
def test_ring_resize_while_pick_in_use_is_safe():
    from upscaler.ring import GpuFrameRing
    ring = GpuFrameRing(capacity=8)
    small = np.full((4, 6, 4), 7, np.uint8)
    for k in range(6):
        ring.push(k * 0.02, small)
    p = ring.pick(0.05)
    assert p is not None
    fa, fb = p.frames()
    big = np.full((8, 12, 4), 9, np.uint8)
    for k in range(6):  # boyut degisti: tampon yeniden ayrildi
        ring.push(1.0 + k * 0.02, big)
    # Eski secimin kareleri hala eski boyutta ve icerikte
    assert fa.shape == (4, 6, 4) and int(fa[0, 0, 0]) == 7
    p.release()  # eski nesilden gelen serbest birakma yeni sayimi bozmamali
    assert ring.resets == 1
    assert not ring._in_use
    slots = list(ring._free) + [e.slot for e in ring._entries]
    assert len(slots) == len(set(slots)) <= ring.capacity
    q = ring.pick(1.05)
    assert q is not None and q.frames()[0].shape == (8, 12, 4)
    q.release()


def _engines_exist(h, w):
    from upscaler.models.fused import engine_paths
    p = engine_paths("rt4ksr-x2", h, w)
    return os.path.exists(p["still"]) and os.path.exists(p["rgbsr"])


@cuda
@pytest.mark.skipif(not _engines_exist(1080, 1920), reason="1920x1080 motoru yok")
def test_fused_processor_routes_by_size():
    from upscaler.process import FusedSrProcessor
    proc = FusedSrProcessor("rt4ksr-x2")
    for (h, w), route in (((1080, 1920), "motor"), ((720, 1280), "sigdirma"), ((700, 1200), "sigdirma")):
        a = torch.randint(0, 255, (h, w, 4), dtype=torch.uint8, device="cuda")
        b = torch.randint(0, 255, (h, w, 4), dtype=torch.uint8, device="cuda")
        for alpha in (0.0, 0.5):
            y = proc(a, b, alpha)
            assert y.shape == (1, 3, 2160, 3840)
        assert proc.route.startswith(route), proc.route
    # 700x1200 -> 1851x1080 icerik: sol/sag kenar siyah (SR sifir girdide 1-3 bias verir)
    assert y[:, :, :, :30].max() <= 8 and y[:, :, :, 200:3600].float().mean() > 60


@cuda
def test_fused_processor_without_canvas_falls_back_to_parts():
    from upscaler.process import FusedSrProcessor
    proc = FusedSrProcessor("rt4ksr-x2", out_h=270, out_w=480, canvases=[])
    a = torch.randint(0, 255, (134, 240, 4), dtype=torch.uint8, device="cuda")
    y = proc(a, a, 0.0)
    assert y.shape == (1, 3, 270, 480)
    assert proc.route.startswith("ayri")
