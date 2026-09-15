import numpy as np

from upscaler.testpattern import decode, render


def test_roundtrip_indices():
    for n in (0, 1, 2, 49, 50, 1234, 65535, 1_000_000):
        assert decode(render(n)) == n


def test_black_frame_is_zero_and_gray_is_none():
    black = np.zeros((720, 1280, 4), np.uint8)
    assert decode(black) == 0
    gray = np.full((720, 1280, 4), 128, np.uint8)
    assert decode(gray) is None


def test_too_small_frame_is_none():
    assert decode(np.zeros((10, 10, 4), np.uint8)) is None


def test_consecutive_frames_differ_in_motion_area():
    a, b = render(100), render(101)
    assert np.any(a[200:, :, :3] != b[200:, :, :3])
