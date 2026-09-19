"""AI on isleme (AiAheadProcessor): sentetik karelerle, senkron AiProcessor ile ayni cikisi vermeli."""
import os

import numpy as np
import pytest
import torch

from upscaler.models.ai import blend_engine_path

NAME, BLEND, GAIN, SAT, CON = "ai_v4", 0.75, 1.6, 1.2, 1.05

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists(blend_engine_path(NAME, BLEND, GAIN, sat=SAT, con=CON)),
    reason="GPU ya da motor yok")


def test_ahead_matches_sync():
    from upscaler.process import AiAheadProcessor, AiProcessor
    from upscaler.ring import GpuFrameRing
    kw = dict(blend=BLEND, gain=GAIN, sat=SAT, con=CON)
    sync = AiProcessor(NAME, 1080, 1920, **kw)
    ahead = AiAheadProcessor(NAME, 1080, 1920, **kw)
    assert ahead.async_ahead
    ring = GpuFrameRing(40)
    ahead.attach(ring)
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (1080, 1920, 4), dtype=np.uint8)
    frames = []
    for i in range(24):
        f = np.roll(base, i * 3, axis=1).copy()
        frames.append(f)
        ring.push(i * 0.02, f, meta=i)
    # on isleme: ilk secim karede 0'i gosterir, sonra ileriyi isler
    outs_ahead, outs_sync = {}, {}
    for k in range(20):
        p = ring.pick(ring._entries[k].stamp.t)
        assert p is not None
        idx = (p.a if p.alpha < 0.5 else p.b).stamp.index
        y = ahead.present_pick(p).clone()
        torch.cuda.current_stream().synchronize()
        outs_ahead[idx] = y
        p.release()
        ahead.work_ahead()
        ahead.ai_stream.synchronize()
    # senkron: kare n gelince f[n-1] cikar
    for k in range(21):
        f = torch.from_numpy(frames[k]).cuda()
        y = sync(f, f, 0.0).clone()
        if k >= 1:
            outs_sync[ring._entries[k - 1].stamp.index] = y
    assert ahead.hits >= 15, (ahead.hits, ahead.misses)
    common = [i for i in outs_ahead if i in outs_sync and i >= ring._entries[2].stamp.index]
    assert len(common) >= 10
    for i in common:
        d = (outs_ahead[i].float() - outs_sync[i].float()).abs().mean().item()
        # ilk pencere farkli baslar (senkron yol ilk kareyi ikiler); 3. kareden sonra ayni olmali
        assert d < 0.5, (i, d)
