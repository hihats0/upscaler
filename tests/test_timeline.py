"""Zaman cizgisi (F23) testleri.

Sorun: Chrome 50 FPS videoyu 144 Hz ekrana basar. Yakalama zamanlari vsync
izgarasina (6.94 ms) yapisir ve titrer. Ara kareler dogru zamana dussun diye
her yeni kareye ideal (esit aralikli) bir zaman damgasi vermeliyiz.
"""
import random

import pytest

from upscaler.timeline import SourceClock, pick_frames


def vsync_quantize(t, hz=144.0, latency=0.004, jitter=0.0015, rng=None):
    """Gercek kare zamanini, ekranin bir sonraki vsync'ine yapistirir (+ yakalama gecikmesi)."""
    rng = rng or random.Random(0)
    period = 1.0 / hz
    k = int(t / period) + 1
    return k * period + latency + rng.uniform(0, jitter)


def feed(clock, times):
    return [clock.push(t) for t in times]


# --- SourceClock -------------------------------------------------------------

def test_perfect_50fps_period_and_indices():
    clock = SourceClock()
    stamps = feed(clock, [n / 50 for n in range(100)])
    assert clock.period == pytest.approx(1 / 50)
    assert [s.index for s in stamps] == list(range(100))


def test_50fps_on_144hz_vsync_is_smoothed_below_1ms():
    rng = random.Random(1)
    clock = SourceClock()
    true_t = [10.0 + n / 50 for n in range(300)]
    stamps = feed(clock, [vsync_quantize(t, rng=rng) for t in true_t])
    assert clock.period == pytest.approx(1 / 50, rel=1e-3)
    # Sabit gecikme onemli degil (ses de ayni gecikmeyle kayar), ARALIKLAR esit olmali.
    tail = stamps[100:]
    gaps = [b.t - a.t for a, b in zip(tail, tail[1:])]
    assert max(abs(g - 1 / 50) for g in gaps) < 0.001
    assert [s.index for s in tail] == list(range(100, 300))


def test_25fps_on_60hz_detected():
    rng = random.Random(2)
    clock = SourceClock()
    feed(clock, [vsync_quantize(5.0 + n / 25, hz=60.0, rng=rng) for n in range(120)])
    assert clock.period == pytest.approx(1 / 25, rel=1e-3)


def test_dropped_frame_advances_index_by_two():
    clock = SourceClock()
    times = [n / 50 for n in range(120)]
    del times[80]  # 80. kare hic gelmedi (saat 1 sn isinmadan sonra kilitli)
    stamps = feed(clock, times)
    assert stamps[80].index == 81
    assert stamps[80].t == pytest.approx(81 / 50, abs=1e-6)


def test_random_drops_do_not_fool_period_on_144hz():
    rng = random.Random(3)
    clock = SourceClock()
    kept = [n for n in range(400) if rng.random() > 0.09]  # ~%9 dusen kare
    stamps = feed(clock, [vsync_quantize(3.0 + n / 50, rng=rng) for n in kept])
    assert clock.period == pytest.approx(1 / 50, rel=1e-3)
    tail = list(zip(kept, stamps))[150:]
    base = tail[0][1].index - tail[0][0]
    assert all(s.index - base == n for n, s in tail)


def test_long_gap_resets_and_flags_discontinuity():
    clock = SourceClock()
    feed(clock, [n / 50 for n in range(60)])
    s = clock.push(60 / 50 + 2.0)  # 2 sn duraklama (ABR, reklam, takilma)
    assert s.discontinuity is True
    s2 = clock.push(60 / 50 + 2.0 + 1 / 50)
    assert s2.discontinuity is False
    assert s2.index == s.index + 1


def test_first_stamps_before_warmup_are_usable():
    clock = SourceClock()
    s0 = clock.push(1.000)
    s1 = clock.push(1.020)
    assert s0.index == 0 and s1.index == 1
    assert s1.t > s0.t


def test_extra_window_updates_do_not_break_the_grid():
    """TOD penceresinde video disi degisiklikler (kontroller, imlec) fazladan kare uretir."""
    rng = random.Random(5)
    events = [(3.0 + n / 50, True, n) for n in range(500)]
    events += [(3.0 + rng.uniform(0, 10.0), False, None) for _ in range(90)]  # ~%18 fazladan
    raws = sorted((vsync_quantize(t, rng=rng), real, n) for t, real, n in events)
    clock = SourceClock()
    out = [(clock.push(r), real, n) for r, real, n in raws]
    assert clock.period == pytest.approx(1 / 50, rel=1e-6)
    locked = [(s, real, n) for s, real, n in out if s.raw > 5.0]
    # Fazladan guncelleme bir sonraki kareye yakin gelirse sirayi o alir, arkadan gelen
    # gercek kare "tekrar" olur. Her iki durumda da gercek karenin SIRASI dogru olmali.
    real = [(s, n) for s, r, n in locked if r]
    base = real[0][0].index - real[0][1]
    wrong = sum(1 for s, n in real if s.index - base != n)
    assert wrong <= 3
    extras_locked = sum(1 for _, r, _ in locked if not r)
    repeats_locked = sum(1 for s, _, _ in locked if s.repeat)
    assert repeats_locked >= 0.8 * extras_locked


def test_ideal_restamps_warmup_frames_on_the_locked_grid():
    rng = random.Random(4)
    clock = SourceClock()
    raws = [vsync_quantize(2.0 + n / 50, rng=rng) for n in range(120)]
    stamps = feed(clock, raws)
    assert clock.ideal(-1.0) is not None
    # Isinmadaki ilk kareler titrek damgalandi; ideal() onlari izgaraya oturtur.
    fixed = [clock.ideal(r) for r in raws[:40]]
    assert [i for i, _ in fixed] == [s.index for s in stamps[:40]]
    gaps = [b[1] - a[1] for a, b in zip(fixed, fixed[1:])]
    assert max(abs(g - 1 / 50) for g in gaps) < 1e-9
    # Kilit sonrasi damgalarla ayni izgarada (stamps[39] isinmada titrek damgalanmisti,
    # o yuzden kilitli son kareden geri sayarak karsilastiriyoruz)
    assert fixed[-1][1] == pytest.approx(stamps[119].t - (119 - 39) / 50, abs=0.001)


def test_ideal_is_none_before_lock():
    clock = SourceClock()
    clock.push(1.0)
    assert clock.ideal(1.0) is None


# --- pick_frames ---------------------------------------------------------------

def test_pick_exact_and_mid():
    ts = [0.00, 0.02, 0.04]
    assert pick_frames(ts, 0.02) == (1, 2, pytest.approx(0.0))
    i, j, a = pick_frames(ts, 0.03)
    assert (i, j) == (1, 2) and a == pytest.approx(0.5)


def test_pick_before_first_returns_none():
    assert pick_frames([1.0, 1.02], 0.9) is None


def test_pick_after_last_holds_last_frame():
    assert pick_frames([1.0, 1.02], 1.5) == (1, 1, 0.0)


def test_50_to_60_alphas_are_multiples_of_one_sixth():
    ts = [n / 50 for n in range(200)]
    alphas = set()
    for k in range(60, 180):
        i, j, a = pick_frames(ts, k / 60)
        alphas.add(round(a * 6, 6))
    assert alphas == {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}
