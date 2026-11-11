# 02 — Publishing the model and an in-browser demo on the Hugging Face Hub

The trained model lives on the Hub, not in this repository, and the demo runs the model inside the
visitor's browser. This guide explains what gets published, how, and why the demo ended up there.

## Concepts

- A **model repository** on the Hub is a git repository with large-file storage. For this model it
  holds:
  - `model.safetensors`: the PyTorch weights. Safetensors only stores tensors, so loading it cannot
    run code, unlike a pickled `.pth`;
  - `config.json`: the architecture plus `id2label`/`label2id`, which is why the Hub shows words
    like "yes" instead of `LABEL_34`;
  - `preprocessor_config.json`: how to normalise audio before the model sees it;
  - `onnx/model.onnx` and `onnx/model_quantized.onnx`: the same network exported to
    [ONNX](https://onnx.ai), a portable graph format that ONNX Runtime can run anywhere, including a
    browser through WebAssembly;
  - `README.md`: the **model card**, with YAML metadata at the top (license, dataset, metrics) that
    the Hub uses for search and for the results widget.
- **Transformers.js** is the JavaScript port of the `pipeline` API. Given a model id, it downloads
  `config.json`, the preprocessor config and an ONNX file, and runs them in the browser.
- **Dynamic quantization** stores weights as 8-bit integers instead of 32-bit floats: about 4x
  smaller, usually with a small accuracy cost. "Dynamic" means activations are quantized on the fly,
  so no calibration data is needed.
- A **Space** is a small app hosted by the Hub. A **static** Space only serves files (HTML, JS, CSS),
  so all the computing happens on the visitor's machine.

## Walkthrough

```bash
hf auth whoami                                   # check the login
make export-onnx                                 # writes runs/wav2vec2/model/onnx/

M=diegomponte/wav2vec2-base-speech-commands-35
hf repos create $M --type model --public
hf upload $M runs/wav2vec2/model . --exclude "training_args.bin"   # weights, configs, onnx/
hf upload $M hub/README.md README.md                                # model card

hf repos create diegomponte/speech-commands-demo --type space --sdk static --public
hf upload diegomponte/speech-commands-demo space . --type space --exclude "*.test.mjs"
```

The page in `space/`:

1. loads the model with `pipeline("audio-classification", MODEL_ID, { dtype: "q8" })`, where `q8`
   selects `onnx/model_quantized.onnx`;
2. records two seconds from the microphone with `MediaRecorder`, or takes an uploaded file;
3. decodes the audio with an `AudioContext` created at 16 kHz, which resamples it for free;
4. keeps the loudest second (`loudestWindow` in `space/audio.js`, tested with `node --test`) and
   shows the five best guesses.

Check that the published weights are the ones that were measured, from anywhere:

```bash
uv run python -m speech_commands.evaluate --data ~/datasets/speech_commands \
    --model diegomponte/wav2vec2-base-speech-commands-35 --out /tmp/check.json
```

## Why

- **Why not commit the weights to GitHub?** They are ~380 MB. The Hub is built for this: large-file
  storage, a download counter and a stable id for `from_pretrained`.
- **Why a static Space instead of a Gradio app?** The first plan was a Gradio Space, but hosting
  Gradio and Docker Spaces now requires a paid plan, while static Spaces stay free. Running the
  model in the browser also means no server to keep awake and audio that never leaves the
  visitor's device. The cost is a ~120 MB download on the first visit (the browser caches it).
- **Why quantize only the MatMuls?** Quantizing everything gave a 95 MB file but test accuracy fell
  from 97.7% to 78.4%. Wav2Vec2's first convolutions read raw audio, and 8-bit weights there lose
  too much detail. Quantizing only the transformer's matrix multiplications gives 122 MB at 97.71%.
  **Always measure a quantized model on the test set**: the export succeeds either way.
- **Why write limitations in the model card?** People decide whether to use a model from its card.
  "English only, single words, one second at 16 kHz, no unknown class" saves someone from a wrong
  assumption.

## Exercises

1. Export an fp16 ONNX model and compare its size, accuracy and speed in the browser with the int8
   one.
2. Pin a model revision in `space/app.js` (`{ revision: "<commit>" }`) and explain when that is
   useful.
3. Open the browser's network tab on the Space and find which files Transformers.js downloads.

## Pitfalls

- **Licenses travel with the model.** The base model is Apache-2.0 and the dataset is CC-BY-4.0, so
  the card credits both, and so does the Space for the example clips.
- **Microphone access needs HTTPS** (or localhost). Spaces are served over HTTPS, so it works there.
- **Pin the Transformers.js version** in the import URL. An unpinned CDN import can change under you.

## Links

- [Model cards](https://huggingface.co/docs/hub/model-cards)
- [Static Spaces](https://huggingface.co/docs/hub/spaces-sdks-static)
- [Transformers.js](https://huggingface.co/docs/transformers.js)
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
