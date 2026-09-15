"""Saat kilidi: gercek kayitlarda periyot dogru kare hizina kilitlenmeli (sadece zaman sayilari)."""
import csv
import os

import pytest

from upscaler.timeline import SourceClock

DATA = os.path.join(os.path.dirname(__file__), "data")


def _replay(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        raws = [float(r["raw_s"]) for r in csv.DictReader(f)]
    clock = SourceClock()
    first = relock_s = None
    for raw in raws:
        before = clock.relocks
        clock.push(raw)
        if clock.period is not None and first is None:
            first = clock.period
        if clock.relocks != before and relock_s is None:
            relock_s = raw - raws[0]
    return clock, first, relock_s


@pytest.mark.parametrize("name", ["tod_window_144hz.csv", "pattern_144hz.csv", "live_pattern_fail1.csv"])
def test_clean_144hz_recordings_lock_once_to_50(name):
    clock, first, _ = _replay(name)
    assert first == pytest.approx(0.02)
    assert clock.period == pytest.approx(0.02)
    assert clock.relocks == 0


def test_48hz_trap_is_corrected_by_verification():
    """TOD, onizleme acik: 1 sn isinma 48 Hz'e kilitliyor, dogrulama birkac saniyede 50'ye cekmeli."""
    clock, first, relock_s = _replay("tod_window_144hz_48hz_trap.csv")
    assert first == pytest.approx(1 / 48)
    assert clock.period == pytest.approx(0.02)
    assert clock.relocks == 1
    assert relock_s is not None and relock_s < 6.0
