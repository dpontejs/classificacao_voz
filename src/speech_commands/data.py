"""Speech Commands v0.02 on disk: labels, official splits and a PyTorch dataset."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 16_000
CLIP_SAMPLES = SAMPLE_RATE  # every clip is at most one second long


def list_labels(root: str | Path) -> list[str]:
    """The 35 command words: every folder except `_background_noise_`, sorted."""
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith("_"))


def _read_list(root: Path, name: str) -> list[str]:
    return [line.strip() for line in (root / name).read_text().splitlines() if line.strip()]


def split_files(root: str | Path) -> dict[str, list[str]]:
    """Official split: validation and test come from the lists shipped with the dataset,
    training is every other clip. Paths are relative, like `yes/0a7c2a8d_nohash_0.wav`."""
    root = Path(root)
    validation = _read_list(root, "validation_list.txt")
    testing = _read_list(root, "testing_list.txt")
    held_out = set(validation) | set(testing)
    everything = sorted(
        f"{wav.parent.name}/{wav.name}"
        for label in list_labels(root)
        for wav in (root / label).glob("*.wav")
    )
    training = [f for f in everything if f not in held_out]
    return {"training": training, "validation": validation, "testing": testing}


def fix_length(wave: np.ndarray, length: int = CLIP_SAMPLES) -> np.ndarray:
    """Zero-pad short clips and crop long ones so every batch has the same shape."""
    if len(wave) >= length:
        return wave[:length]
    return np.pad(wave, (0, length - len(wave)))


class SpeechCommandsDataset(Dataset):
    """Reads one clip per item and turns it into Wav2Vec2 input values plus a label index."""

    def __init__(self, root, files: list[str], labels: list[str], feature_extractor):
        self.root = Path(root)
        self.files = files
        self.label_ids = {label: i for i, label in enumerate(labels)}
        self.feature_extractor = feature_extractor

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict:
        path = self.files[i]
        wave, rate = sf.read(self.root / path, dtype="float32")
        assert rate == SAMPLE_RATE, f"{path}: expected {SAMPLE_RATE} Hz, got {rate}"
        features = self.feature_extractor(fix_length(wave), sampling_rate=SAMPLE_RATE)
        return {
            "input_values": torch.as_tensor(np.asarray(features["input_values"][0])),
            "labels": self.label_ids[path.split("/")[0]],
        }
