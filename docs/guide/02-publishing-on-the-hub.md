# 02 — Publishing the model and a demo on the Hugging Face Hub

The trained model lives on the Hub, not in this repository. This guide explains what gets published
and how.

## Concepts

- A **model repository** on the Hub is a git repository with large-file storage. For Transformers
  models it holds:
  - `model.safetensors`: the weights. Safetensors only stores tensors, so loading it cannot run
    code, unlike a pickled `.pth`;
  - `config.json`: the architecture plus `id2label`/`label2id`, which is why the Hub shows words
    like "yes" instead of `LABEL_34`;
  - `preprocessor_config.json`: how to normalise audio before the model sees it;
  - `README.md`: the **model card**, with YAML metadata at the top (license, dataset, metrics)
    that the Hub uses for search and for the results widget.
- A **Space** is a small app hosted by the Hub. `space/` holds a Gradio app that loads the model by
  its id, so the demo always uses the published weights.

## Walkthrough

```bash
hf auth whoami                                   # check the login
hf repos create diegomponte/wav2vec2-base-speech-commands-35 --type model
hf upload diegomponte/wav2vec2-base-speech-commands-35 runs/wav2vec2/model . \
    --exclude "training_args.bin"
hf upload diegomponte/wav2vec2-base-speech-commands-35 hub/README.md README.md

hf repos create diegomponte/speech-commands-demo --type space --sdk gradio
hf upload diegomponte/speech-commands-demo space . --type space
```

Then check that the model works from anywhere:

```python
from transformers import pipeline

classifier = pipeline("audio-classification", model="diegomponte/wav2vec2-base-speech-commands-35")
classifier("my_recording.wav")
```

## Why

- **Why not commit the weights to GitHub?** They are ~380 MB. The Hub is built for this: large-file
  storage, a download counter, the inference widget and a stable id for `from_pretrained`.
- **Why write limitations in the model card?** People decide whether to use a model from its card.
  "English only, single words, one second at 16 kHz" saves someone from a wrong assumption.
- **Why a separate Space?** The model repository stays a clean artifact. The Space can change
  (UI, preprocessing) without new model versions.

## Exercises

1. Add a `hub/README.md` metadata field for the validation accuracy and see how the Hub renders it.
2. Pin a model revision in `space/app.py` (`revision="<commit>"`) and explain when that is useful.

## Pitfalls

- **Licenses travel with the model.** The base model is Apache-2.0 and the dataset is CC-BY-4.0, so
  the card credits both.
- **Space hardware.** The free CPU tier is enough for a 95M-parameter model on one-second clips, but
  the first request after the Space sleeps takes a few seconds to load the model.

## Links

- [Model cards](https://huggingface.co/docs/hub/model-cards)
- [Gradio Spaces](https://huggingface.co/docs/hub/spaces-sdks-gradio)
- [Safetensors](https://huggingface.co/docs/safetensors)
