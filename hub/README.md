---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: audio-classification
base_model: facebook/wav2vec2-base
datasets:
  - google/speech_commands
tags:
  - audio
  - keyword-spotting
  - speech-commands
  - wav2vec2
model-index:
  - name: wav2vec2-base-speech-commands-35
    results:
      - task:
          type: audio-classification
          name: Audio Classification
        dataset:
          type: google/speech_commands
          name: Speech Commands v0.02 (35 words, official test split)
          config: v0.02
          split: test
        metrics:
          - type: accuracy
            value: 0.9772
          - type: f1
            name: Macro F1
            value: 0.9752
---

# Wav2Vec2-base for 35 spoken commands

`facebook/wav2vec2-base` fine-tuned on [Speech Commands v0.02](https://arxiv.org/abs/1804.03209) to
recognise 35 spoken English words: yes, no, up, down, left, right, on, off, stop, go, forward,
backward, follow, learn, the digits zero to nine, and bed, bird, cat, dog, happy, house, marvin,
sheila, tree, visual, wow.

**Try it:** [speech-commands-demo](https://huggingface.co/spaces/diegomponte/speech-commands-demo) ·
**Code:** [github.com/dpontejs/classificacao_voz](https://github.com/dpontejs/classificacao_voz)

## Usage

```python
from transformers import pipeline

classifier = pipeline("audio-classification", model="diegomponte/wav2vec2-base-speech-commands-35")
classifier("my_recording.wav", top_k=3)
# [{'label': 'yes', 'score': 0.99}, ...]
```

Input should be **one second of 16 kHz mono audio** with a single word. For longer recordings, cut
the second around the word first (the demo keeps the loudest second).

## Results

Official test split: 11,005 clips from speakers never seen in training.

| Metric | Value |
|---|---|
| Accuracy | **97.72%** |
| Macro F1 | 0.975 |
| Validation accuracy (best epoch) | 97.75% |

Hardest words by F1: "tree" (0.91), "forward" (0.95), "learn" (0.95). Most errors are between words
that sound alike: "tree" ↔ "three" (25 of the errors) and "off" heard as "up" (10).

This is the **35-word** task. It is not comparable to the 12-class keyword-spotting benchmark
(10 words + "unknown" + "silence") reported by SUPERB and many papers.

## Training

- **Data:** the full training split (84,843 clips) with the official validation and test lists, so
  no speaker appears in two splits. Clips are zero-padded to one second and normalised by the
  Wav2Vec2 feature extractor.
- **Model:** `Wav2Vec2ForSequenceClassification` (projection to 256, mean pooling over time, linear
  classifier). The convolutional feature encoder is frozen; the transformer and head are trained.
- **Hyperparameters:** AdamW, learning rate 3e-5, 10% linear warmup, batch 32, fp16, 5 epochs,
  seed 42. Best epoch chosen by validation accuracy.
- **Hardware:** one RTX 4050 laptop GPU (6 GB), 35 minutes.

## Limitations

- English only, single isolated words, one second long. It does not transcribe speech.
- It always picks one of the 35 words: there is no "unknown" or "silence" class, so any other sound
  still gets a label. Use the score as a confidence threshold if that matters.
- Speech Commands was recorded by volunteers through browsers and phones; accents, children's
  voices and noisy rooms are less represented than in real use.

## License and credits

Apache-2.0, like the base model [`facebook/wav2vec2-base`](https://huggingface.co/facebook/wav2vec2-base).
Trained on Speech Commands by Pete Warden, released under CC-BY-4.0.
