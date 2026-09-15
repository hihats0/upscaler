"""schedule.select: donma boslugunda ara kare uretilmez, eski kare tutulur."""
import numpy as np
import pytest
import torch

from upscaler import schedule
from upscaler.ring import GpuFrameRing

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")


def _ring_with_gap():
    ring = GpuFrameRing(capacity=200)
    f = np.zeros((4, 4, 4), np.uint8)
    for k in range(100):          # 2 sn, 50 FPS
        ring.push(10.0 + k * 0.02, f)
    for k in range(50):           # 4 sn donma, sonra yayin devam
        ring.push(16.0 + k * 0.02, f)
    return ring


def test_gap_holds_last_frame_no_interpolation():
    ring = _ring_with_gap()
    p, s = schedule.select(ring, 13.0, 60.0)
    assert p is not None
    assert p.alpha == 0.0
    assert p.b.stamp.index != p.a.stamp.index
    assert abs(p.a.stamp.t - 11.98) < 0.03  # donmadan onceki son kare
    p.release()


def test_normal_region_interpolates():
    ring = _ring_with_gap()
    p, s = schedule.select(ring, 11.0 + 1 / 60, 60.0)
    assert p is not None and p.b.stamp.index == p.a.stamp.index + 1
    assert 0.0 < p.alpha < 1.0
    p.release()
