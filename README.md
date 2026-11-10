# Speech Command Recognition with Wav2Vec2

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-F97316?style=flat-square&logo=gradio&logoColor=white)

Recognises 35 spoken English commands ("yes", "no", "up", "down", digits, …) by fine-tuning
**Wav2Vec2** on the full [Speech Commands v0.02](https://arxiv.org/abs/1804.03209) dataset.

- **Model:** [diegomponte/wav2vec2-base-speech-commands-35](https://huggingface.co/diegomponte/wav2vec2-base-speech-commands-35)
- **Live demo:** [diegomponte/speech-commands-demo](https://huggingface.co/spaces/diegomponte/speech-commands-demo)

```python
from transformers import pipeline

classifier = pipeline("audio-classification", model="diegomponte/wav2vec2-base-speech-commands-35")
classifier("my_recording.wav")  # one second, 16 kHz
```

## Results

Official test split (11,005 clips, speakers never seen in training):

| Model | Training data | Accuracy | Macro F1 |
|---|---|---|---|
| **This version** | full training split (84,843 clips) | **97.72%** | **0.975** |
| First version (`notebooks/`) | 10% subset | 91.44% | 0.901 |

Most mistakes are between words that sound alike: "tree" and "three" account for 25 of them,
then "off" heard as "up". Per-class F1 and the confusion matrix are in
[`results/test_metrics.json`](results/test_metrics.json).

## How it works

- **Data:** the official split lists, so no speaker appears in two splits. Every clip is padded to
  one second and normalised by the Wav2Vec2 feature extractor.
- **Model:** `facebook/wav2vec2-base` with an audio-classification head (a projection, mean pooling
  over time and a linear classifier). The convolutional feature encoder is frozen; the transformer
  and the head are trained.
- **Training:** Hugging Face `Trainer`, AdamW, learning rate 3e-5 with 10% warmup, batch 32, fp16,
  5 epochs. The best epoch by validation accuracy is kept, and the test split is used once.

The guides in [`docs/guide/`](docs/guide/) explain each step and why:

1. [Fine-tuning Wav2Vec2 for spoken commands](docs/guide/01-fine-tuning-wav2vec2.md)
2. [Publishing the model and a demo on the Hugging Face Hub](docs/guide/02-publishing-on-the-hub.md)

## Running

Needs [uv](https://docs.astral.sh/uv/) and, for training, a CUDA GPU (6 GB is enough).

```bash
mkdir -p ~/datasets/speech_commands && cd ~/datasets/speech_commands
curl -L http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz | tar -xz
cd -

make test       # unit tests
make train      # ~35 min on an RTX 4050 laptop GPU
make evaluate   # writes results/test_metrics.json
```

| Path | Contents |
|---|---|
| `src/speech_commands/` | Data loading, training, evaluation |
| `space/` | Gradio demo deployed as a Hugging Face Space |
| `results/` | Test metrics of the published model |
| `notebooks/` | The first version, trained on Kaggle (kept for history) |

## History

The first version (`notebooks/`) trained on a 10% subset to fit a Kaggle GPU budget and reported
95.37% on 1,101 test clips. That checkpoint was not kept; the one that was scores 92.92% on the same
clips and 91.44% on the full test split. This version retrains on the full dataset, publishes the
weights together with their metrics, and documents why in
[guide 01](docs/guide/01-fine-tuning-wav2vec2.md#a-lesson-in-reproducibility).

## License

Code under Apache-2.0. The model is derived from `facebook/wav2vec2-base` (Apache-2.0) and trained on
Speech Commands (CC-BY-4.0).
