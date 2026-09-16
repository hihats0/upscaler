"""TOD simulatoru kare numarasi seridi (sentetik; ffmpeg gerekmez)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tod_sim  # noqa: E402


def draw(idx: int) -> np.ndarray:
    img = np.full((1080, 1920, 4), 90, np.uint8)
    img[: tod_sim.STRIP_H, : (tod_sim.BITS + tod_sim.CHECK_BITS + 2) * tod_sim.CELL_W] = 0
    bits = [(idx >> b) & 1 for b in range(tod_sim.BITS)]
    chk = sum(bits) % 16
    bits += [(chk >> c) & 1 for c in range(tod_sim.CHECK_BITS)]
    for b, v in enumerate(bits):
        if v:
            x = (b + 1) * tod_sim.CELL_W
            img[: tod_sim.STRIP_H, x:x + tod_sim.CELL_W] = 255
    return img


def test_barcode_roundtrip():
    for idx in (0, 1, 5, 499, 12345, 2 ** 20 - 1):
        assert tod_sim.read_barcode(draw(idx)) == idx


def test_barcode_rejects_bad_check():
    img = draw(37)
    x = (1 + 3) * tod_sim.CELL_W  # bir veri bitini boz
    img[: tod_sim.STRIP_H, x:x + tod_sim.CELL_W] ^= 255
    assert tod_sim.read_barcode(img) is None


def test_filter_mentions_all_cells():
    f = tod_sim.barcode_filter()
    assert f.count("drawbox") == 1 + tod_sim.BITS + tod_sim.CHECK_BITS
