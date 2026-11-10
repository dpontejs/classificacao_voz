"""Turn a recording of any length into the one-second clip the model was trained on."""

import numpy as np

SAMPLE_RATE = 16_000


def loudest_window(wave: np.ndarray, length: int = SAMPLE_RATE) -> np.ndarray:
    """Zero-pad short recordings; for long ones, keep the window with the most energy,
    which is where the spoken command is when someone records a few seconds."""
    if len(wave) <= length:
        return np.pad(wave, (0, length - len(wave)))
    energy = np.concatenate([[0.0], np.cumsum(wave.astype("float64") ** 2)])
    window_energy = energy[length:] - energy[:-length]
    start = int(np.argmax(window_energy))
    return wave[start : start + length]
