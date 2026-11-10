# 01 — Fine-tuning Wav2Vec2 for spoken commands

This guide explains how `src/speech_commands/` turns a pretrained speech model into a classifier
for 35 spoken words, and why each choice was made.

## Concepts

- **Wav2Vec2** learns speech representations from raw audio without labels (self-supervised
  learning on 960 h of LibriSpeech). It has two parts:
  - a **convolutional feature encoder**, which turns 16 kHz samples into one feature vector every
    20 ms (a 1 s clip becomes 49 frames);
  - a **transformer** of 12 layers, which turns those frames into contextual representations.
- **Fine-tuning for classification** adds a small head on top: project each frame to 256
  dimensions, average the frames over time (mean pooling) and output one score per word.
  `AutoModelForAudioClassification` builds exactly that, so the result works with
  `pipeline("audio-classification")` with no custom code.
- **Speech Commands v0.02** has ~105k one-second clips of 35 words from thousands of speakers. It
  ships with `validation_list.txt` and `testing_list.txt`, chosen so that **no speaker appears in
  two splits**. Splitting at random instead would leak speakers into the test set and inflate
  the numbers.

## Walkthrough

1. `data.list_labels` lists the 35 word folders, skipping `_background_noise_`.
2. `data.split_files` reads the two official lists; training is every other clip.
3. `data.SpeechCommandsDataset` reads a clip, pads it to exactly one second (`fix_length`), and runs
   the `Wav2Vec2FeatureExtractor`, which normalises each clip to zero mean and unit variance.
4. `train.py` loads `facebook/wav2vec2-base`, **freezes the feature encoder** and fine-tunes the
   transformer and the head for 5 epochs with the Hugging Face `Trainer`:
   AdamW, learning rate 3e-5 with 10% warmup, batch 32, mixed precision (fp16).
5. The best epoch by **validation** accuracy is kept. The test split is used once, by
   `evaluate.py`, after training is over.

```bash
make train      # ~35 min on a laptop RTX 4050 (6 GB)
make evaluate   # writes results/test_metrics.json
```

## Why

- **Why freeze the feature encoder?** Its convolutions learned generic acoustic features that
  transfer well, and it is the part that uses most of the memory. Freezing it is the standard
  recipe from the Wav2Vec2 paper, and it lets training fit in 6 GB.
- **Why pad to exactly one second?** Every batch gets the same shape, and the model sees at
  inference time what it saw in training. The demo does the same by keeping the loudest second of
  a recording.
- **Why the full dataset?** The first version (the notebooks in `notebooks/`) trained on 10% of the
  data to fit a Kaggle GPU budget. With a local GPU there is no reason to throw data away.

## A lesson in reproducibility

The first version reported 95.37% test accuracy. When the checkpoint was evaluated again, on the
same 1,101 clips, it scored 92.92%. The data matched exactly (same number of clips per class) and
library versions made no difference, so the checkpoint that was kept was simply not the one the
notebook measured: the notebook's output, where the real one lived, was never saved.

What this version does differently:

- the code lives in a repository, not in a notebook session;
- the model and its metrics are published together on the Hub, so the numbers in the model card
  come from the exact weights people download;
- `results/test_metrics.json` is committed, and anyone can rerun `make evaluate` against the Hub
  model (`--model diegomponte/wav2vec2-base-speech-commands-35`) to check it.

## Exercises

1. Train with the transformer frozen too (only the head learns). How much accuracy is lost, and how
   much faster is each epoch?
2. Plot the confusion matrix from `results/test_metrics.json`. Which pairs of words get confused,
   and do they sound alike?
3. Add background noise from `_background_noise_` as data augmentation, and measure the effect on
   your own microphone recordings in the demo.

## Pitfalls

- **Sampling rate.** The model expects 16 kHz. A 44.1 kHz recording fed without resampling gives
  confident nonsense. The demo resamples with `librosa`.
- **Speaker leakage.** Never build your own random split of this dataset; use the official lists.
- **This is not the 12-class keyword-spotting benchmark** (10 words + "unknown" + "silence") that
  many papers report. Compare numbers only with results on the same 35-class task.

## Links

- [wav2vec 2.0 paper](https://arxiv.org/abs/2006.11477)
- [Speech Commands paper](https://arxiv.org/abs/1804.03209)
- [Audio classification guide in Transformers](https://huggingface.co/docs/transformers/tasks/audio_classification)
