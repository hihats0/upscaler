"""Hat 1.4 paket 2-3 saf yardimcilari: ekran secimi, vsync plani, sunum aralik ozeti,
ses yakalama saati, ses halkasi ve kayma duzeltmeli calma (sahte saatlerle)."""
import math
import random
from types import SimpleNamespace

import numpy as np

from upscaler.audio import FS, AudioRing, CaptureClock, SyncedPlayer
from upscaler.present import MonitorInfo, VsyncClock, fit_rect, pacing_summary, pick_monitor, swap_plan


def test_pick_monitor_prefers_external_4k():
    laptop = MonitorInfo("laptop", 0, 0, 1920, 1080, 144, primary=True)
    tv = MonitorInfo("LG TV", 1920, 0, 3840, 2160, 60)
    assert pick_monitor([laptop, tv]) == 1
    assert pick_monitor([laptop]) == 0
    assert pick_monitor([tv, laptop], prefer="laptop") == 1
    assert pick_monitor([MonitorInfo("4k30", 0, 0, 3840, 2160, 30), laptop]) == 1  # 30 Hz 4K degil
    assert pick_monitor([]) is None


def test_swap_plan():
    assert swap_plan(60, 60) == ("lock", 1)
    assert swap_plan(59.94, 60) == ("lock", 1)
    assert swap_plan(120, 60) == ("lock", 2)
    assert swap_plan(144, 60) == ("timer", 0)
    assert swap_plan(144, 72) == ("lock", 2)
    assert swap_plan(60, 60, "timer") == ("timer", 0)


def test_fit_rect():
    assert fit_rect(3840, 2160, 1920, 1080) == (0, 0, 1920, 1080)
    assert fit_rect(3840, 2160, 1920, 1200) == (0, 60, 1920, 1080)


def test_vsync_clock_tracks_period_and_skips_missed():
    vc = VsyncClock(1 / 60)
    true_p = 1 / 59.94
    t = 100.0
    for k in range(600):
        t += true_p * (2 if k % 97 == 0 else 1)  # ara sira bir vsync kacti
        vc.update(t + random.uniform(-0.0003, 0.0003))
    assert abs(vc.period - true_p) < 2e-5
    nxt = vc.next_vsync(t + 0.001)
    assert abs(nxt - (t + true_p)) < 0.001


def test_pacing_summary_counts_late():
    p = 1 / 60
    ticks = [k * p for k in range(300)]
    ends = [T + 0.004 for T in ticks]
    ends[100] += 0.012  # yarim periyottan fazla gec
    s = pacing_summary(ends, ticks, p)
    assert s["gec_sunum"] == 1
    assert abs(s["sunum_araligi_ms_p1_p50_p99"][1] - 16.67) < 0.05


def test_capture_clock_lower_envelope_and_drift():
    fs_true = FS * (1 + 300e-6)  # ses karti saati perf_counter'dan 300 ppm hizli
    c0 = 50.0
    clock = CaptureClock()
    rng = random.Random(1)
    n = 0
    for k in range(3000):  # 30 sn, 10 ms parcalar
        n += 480
        arrival = c0 + n / fs_true + rng.expovariate(1 / 0.006)  # gelis hep sonra, titrek
        clock.add(n, arrival)
    err_ms = (clock.time_of(n) - (c0 + n / fs_true)) * 1000
    assert abs(err_ms) < 1.5
    assert abs(clock.fs - fs_true) < 3


def test_audio_ring_interp_and_wrap():
    ring = AudioRing(seconds=0.01)  # 480 ornek
    ramp = np.stack([np.arange(1000, dtype=np.float32)] * 2, 1)
    ring.write(ramp[:700])
    ring.write(ramp[700:])
    got = ring.read_interp(np.array([600.25, 999.5, 100.0]))
    assert abs(got[0, 0] - 600.25) < 1e-3
    assert got[1, 0] == 0.0  # son orneğin sonrasi yok
    assert got[2, 0] == 0.0  # tampondan dusmus


def test_synced_player_locks_without_drift():
    """Yakalama 200 ppm hizli, DAC 150 ppm yavas; 5 sn gecikmeyle 3 dk: hata < 1 ms, sert atlama yok."""
    fs_cap = FS * (1 + 200e-6)
    fs_dac = FS * (1 - 150e-6)
    ring = AudioRing(seconds=8)
    clock = CaptureClock()
    player = SyncedPlayer(ring, clock)
    player._pa_offset = 0.0
    player.delay_s = 1.5
    block = 480
    cap_n = 0
    t_cap0 = 10.0
    dac_n = 0
    t_dac0 = 10.0
    out = np.zeros((block, 2), np.float32)
    status = SimpleNamespace(output_underflow=False)
    errs = []
    steps = int(180 * FS / block)
    for k in range(steps):
        # yakalama: bir parca (deger = yakalama zamani * 1000 ms rampasi, kontrol icin)
        t_end = t_cap0 + (cap_n + block) / fs_cap
        x = (t_cap0 + (cap_n + np.arange(block)) / fs_cap).astype(np.float64)
        ring.write(np.stack([(x % 1.0).astype(np.float32)] * 2, 1))
        cap_n += block
        clock.add(cap_n, t_end + 0.002)
        # calma: DAC saatine gore bir blok
        dac = t_dac0 + dac_n / fs_dac
        player._cb(out, block, SimpleNamespace(outputBufferDacTime=dac), status)
        dac_n += block
        if k > steps // 4 and player.stats.silent_blocks < k:
            errs.append(abs(player.stats.err_ms))
            if k % 500 == 0:
                # calinan ornegin yakalama zamani = dac - delay (2 ms gelis ofseti alt zarfta kalir)
                want = (dac - player.delay_s) % 1.0
                if 0.05 < want < 0.95:
                    assert abs(out[0, 0] - want) < 0.004
    assert max(errs) < 1.0
    assert player.stats.hard_jumps == 0


def test_pick_monitor_external_for_tv():
    panel = MonitorInfo("Generic PnP Monitor", 0, 0, 1920, 1080, 144, primary=True)
    tv = MonitorInfo("Generic PnP Monitor", -1920, 0, 1920, 1080, 50)
    assert pick_monitor([panel, tv], prefer="external") == 1
    assert pick_monitor([panel], prefer="external") == 0
    assert swap_plan(50, 50) == ("lock", 1)
    assert swap_plan(144, 50) == ("timer", 0)
