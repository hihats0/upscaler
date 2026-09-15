"""Gercek yakalama zamanlariyla SourceClock tekrar oynatma testleri.

Veri: tests/data/*.csv (sadece sayi: WGC zaman damgasi ve test deseni barkodu).
Goruntu yok. TOD kaydinda barkod yok (-1), sadece karelerin gelis zamanlari var.

Uretmek icin:
    tools/test_pattern_window.py + tools/probe_capture.py --pattern --dump tests/data/pattern_144hz.csv
    tools/probe_capture.py --title "TOD - Google Chrome" --foreground --dump tests/data/tod_window_144hz.csv
"""
import csv
import os

import pytest

from upscaler.timeline import SourceClock

DATA = os.path.join(os.path.dirname(__file__), "data")


def load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        pytest.skip(f"veri yok: {name}")
    with open(path, encoding="utf-8") as f:
        return [(float(r["raw_s"]), int(r["barcode"])) for r in csv.DictReader(f)]


def test_pattern_replay_indices_and_times():
    rows = load("pattern_144hz.csv")
    clock = SourceClock()
    stamps = [(clock.push(raw), code) for raw, code in rows]
    assert clock.period == pytest.approx(0.02, rel=1e-6)
    start = rows[0][0] + 1.5  # isinma sonrasi
    locked = [(s, c) for s, c in stamps if s.raw > start and c >= 0]
    s0, c0 = locked[0]
    base_i = s0.index - c0
    new = [(s, c) for s, c in locked if not s.repeat]
    idx_ok = sum(1 for s, c in new if s.index - base_i == c) / len(new)
    # Ideal zaman, barkodun soyledigi kaynak zamanina ne kadar yakin?
    base_t = s0.t - clock.period * c0
    errs = sorted(abs(s.t - (base_t + clock.period * c)) * 1000 for s, c in new)
    p99 = errs[int(len(errs) * 0.99)]
    print(f"\nkare {len(new)}, dogru sira %{idx_ok * 100:.2f}, zaman hatasi p99 {p99:.2f} ms, max {errs[-1]:.2f} ms,"
          f" tekrar {clock.repeats}")
    assert idx_ok >= 0.995
    assert p99 < 1.0


def test_tod_replay_locks_to_50fps():
    rows = load("tod_window_144hz.csv")
    clock = SourceClock()
    stamps = [clock.push(raw) for raw, _ in rows]
    span = rows[-1][0] - rows[0][0]
    new = sum(1 for s in stamps if not s.repeat)
    print(f"\nTOD: {len(rows)} benzersiz, {new} yeni kare, {clock.repeats} tekrar, "
          f"{span:.1f} sn, periyot {clock.period and clock.period * 1000:.3f} ms")
    assert clock.period == pytest.approx(0.02, rel=1e-6)
    assert new / span == pytest.approx(50, abs=1.0)
