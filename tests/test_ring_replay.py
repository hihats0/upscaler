"""GpuFrameRing tekrar oynatma: gercek yakalama zamanlari + sahte 2x2 kareler.

Saat (SourceClock) tek basina bazi durumlari aninda cozemez (144 Hz'de "yetisen"
kare); tampon karari bir kare erteleyerek cozer. Bu yuzden asil dogruluk olcutu
tampondaki girdilerin zamanidir.
"""
import csv
import os
import random

import numpy as np
import pytest
import torch

from upscaler.ring import GpuFrameRing

DATA = os.path.join(os.path.dirname(__file__), "data")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")


def load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        pytest.skip(f"veri yok: {name}")
    with open(path, encoding="utf-8") as f:
        return [(float(r["raw_s"]), int(r["barcode"])) for r in csv.DictReader(f)]


def replay(rows):
    ring = GpuFrameRing(capacity=4000)
    dummy = np.zeros((2, 2, 4), np.uint8)
    for raw, code in rows:
        ring.push(raw, dummy, code)
    return ring


def entry_errors(ring, skip_until):
    P = ring.clock.period
    ents = [e for e in ring._entries if e.meta is not None and e.meta >= 0 and e.stamp.raw > skip_until]
    base = float(np.median([e.stamp.t - P * e.meta for e in ents]))
    return [abs(e.stamp.t - (base + P * e.meta)) * 1000 for e in ents], ents


@pytest.mark.parametrize("name", ["live_pattern_fail1.csv", "pattern_144hz.csv"])
def test_ring_entries_match_barcodes(name):
    rows = load(name)
    ring = replay(rows)
    errs, ents = entry_errors(ring, rows[0][0] + 1.5)
    codes = [e.meta for e in ents]
    print(f"\n{name}: girdi {len(ents)}, tekrar {ring.repeats}, bosluk doldurma {ring.hole_fills}, "
          f"duzeltme {ring.fixes}, max hata {max(errs):.2f} ms")
    assert codes == sorted(codes) and len(set(codes)) == len(codes)
    assert max(errs) < 2.0


def test_ring_late_then_on_time_synthetic():
    rng = random.Random(6)
    rows = []
    for n in range(400):
        t = (int((4.0 + n / 50) * 144) + 1) / 144 + 0.004 + rng.uniform(0, 0.0015)
        if n > 100 and n % 37 == 0:
            t += 0.012  # bu kare 12 ms gec, arkasindaki zamaninda
        rows.append((t, n))
    rows.sort()
    ring = replay(rows)
    errs, ents = entry_errors(ring, 6.0)
    assert [e.meta for e in ents] == sorted(e.meta for e in ents)
    assert max(errs) < 2.0


def test_ring_tod_window_is_50fps():
    rows = load("tod_window_144hz.csv")
    ring = replay(rows)
    ents = list(ring._entries)
    span = ents[-1].stamp.t - ents[0].stamp.t
    idx = [e.stamp.index for e in ents]
    assert idx == sorted(idx) and len(set(idx)) == len(idx)
    assert ring.clock.period == pytest.approx(0.02, rel=1e-6)
    assert (len(ents) - 1) / span == pytest.approx(50, abs=1.0)
