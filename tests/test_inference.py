import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "space"))
from inference import loudest_window  # noqa: E402


def test_short_recording_is_zero_padded_to_one_second():
    out = loudest_window(np.ones(100, "float32"), length=1000)
    assert out.shape == (1000,)
    assert out[:100].sum() == 100 and out[100:].sum() == 0


def test_long_recording_keeps_the_loudest_second():
    quiet, loud = np.full(3000, 0.01, "float32"), np.full(1000, 0.9, "float32")
    wave = np.concatenate([quiet, loud, quiet])
    out = loudest_window(wave, length=1000)
    assert out.shape == (1000,)
    assert np.allclose(out, 0.9)


def test_exact_length_is_untouched():
    wave = np.arange(1000, dtype="float32")
    assert np.array_equal(loudest_window(wave, length=1000), wave)
