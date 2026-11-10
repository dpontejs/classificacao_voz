import numpy as np
import pytest
import soundfile as sf

from speech_commands.data import (
    SAMPLE_RATE,
    SpeechCommandsDataset,
    fix_length,
    list_labels,
    split_files,
)


@pytest.fixture
def root(tmp_path):
    """A miniature Speech Commands tree: 3 words, noise folder, official split lists."""
    clips = {
        "yes": ["a_nohash_0.wav", "b_nohash_0.wav", "c_nohash_0.wav"],
        "no": ["a_nohash_0.wav", "d_nohash_0.wav"],
        "up": ["e_nohash_0.wav"],
    }
    for word, names in clips.items():
        (tmp_path / word).mkdir()
        for name in names:
            sf.write(tmp_path / word / name, np.zeros(SAMPLE_RATE // 2, "float32"), SAMPLE_RATE)
    (tmp_path / "_background_noise_").mkdir()
    sf.write(tmp_path / "_background_noise_" / "white.wav", np.zeros(10), SAMPLE_RATE)
    (tmp_path / "validation_list.txt").write_text("yes/b_nohash_0.wav\n")
    (tmp_path / "testing_list.txt").write_text("no/d_nohash_0.wav\nyes/c_nohash_0.wav\n")
    return tmp_path


def test_labels_are_sorted_words_without_background_noise(root):
    assert list_labels(root) == ["no", "up", "yes"]


def test_official_lists_define_validation_and_test(root):
    splits = split_files(root)
    assert splits["validation"] == ["yes/b_nohash_0.wav"]
    assert splits["testing"] == ["no/d_nohash_0.wav", "yes/c_nohash_0.wav"]


def test_training_is_everything_else_and_splits_are_disjoint(root):
    splits = split_files(root)
    assert splits["training"] == ["no/a_nohash_0.wav", "up/e_nohash_0.wav", "yes/a_nohash_0.wav"]
    seen = [f for files in splits.values() for f in files]
    assert len(seen) == len(set(seen))


def test_fix_length_pads_short_clips_with_silence():
    out = fix_length(np.ones(10, "float32"), 16)
    assert out.shape == (16,)
    assert out[:10].tolist() == [1.0] * 10
    assert out[10:].tolist() == [0.0] * 6


def test_fix_length_crops_long_clips():
    assert fix_length(np.arange(20, dtype="float32"), 16).tolist() == list(range(16))


class FakeExtractor:
    """Stands in for Wav2Vec2FeatureExtractor: records the call, returns the wave."""

    def __call__(self, wave, sampling_rate):
        assert sampling_rate == SAMPLE_RATE
        return {"input_values": [wave * 2]}


def test_dataset_yields_one_second_features_and_label_index(root):
    labels = list_labels(root)
    ds = SpeechCommandsDataset(root, ["up/e_nohash_0.wav"], labels, FakeExtractor())
    item = ds[0]
    assert len(ds) == 1
    assert item["labels"] == labels.index("up")
    assert item["input_values"].shape == (SAMPLE_RATE,)
