"""Canli uctan uca test: 50 FPS test deseni -> WGC -> tampon -> 60 Hz -> 4K.

Calistirma (yavas, gercek pencere ve GPU ister):
    .venv/Scripts/python.exe -m pytest -m live -q
"""
import os
import subprocess
import sys
import time

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA yok")]


@pytest.fixture
def pattern_window():
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "tools", "test_pattern_window.py"),
                             "--fps", "50", "--seconds", "20"], stdout=subprocess.PIPE, text=True)
    time.sleep(2.5)
    yield proc
    proc.terminate()
    proc.wait(timeout=5)


def test_pattern_through_live_pipeline(pattern_window):
    from upscaler.live import LiveConfig, run

    s = run(LiveConfig(title="upscaler-test-pattern", delay=1.0, seconds=9.0,
                       preview=False, pattern=True, quiet=True,
                       dump_timing=os.path.join(ROOT, "tests", "data", "live_pattern_last.csv")))
    print(s)
    assert s["kaynak"] == "1280x720"
    assert s["giris_fps"] == pytest.approx(50, abs=1.5)
    assert s["saat_periyot_ms"] == pytest.approx(20.0, abs=0.01)
    assert s["cikis_fps_aktif"] >= 58.5
    assert s["gec_tik"] <= 5
    assert s["tutma_tik"] == 0
    p50, p99, mx = s["cikis_zaman_hatasi_ms_p50_p99_max"]
    assert p99 < 2.0, "cikis karesi yanlis kaynak zamanina dusuyor"
