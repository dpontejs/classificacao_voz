---
title: Speech Commands Demo
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: apache-2.0
models:
  - diegomponte/wav2vec2-base-speech-commands-35
short_description: Say one English word; Wav2Vec2 guesses which of 35 it is
---

# Speech command recognition

Record or upload one spoken English word and the model returns its five best guesses among the 35
words of Speech Commands v0.02.

- Model: [diegomponte/wav2vec2-base-speech-commands-35](https://huggingface.co/diegomponte/wav2vec2-base-speech-commands-35)
- Code: [github.com/dpontejs/classificacao_voz](https://github.com/dpontejs/classificacao_voz)
