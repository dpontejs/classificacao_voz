"""Gradio demo: record or upload a spoken command and see the model's top guesses."""

import os

import gradio as gr
import librosa
from inference import SAMPLE_RATE, loudest_window
from transformers import pipeline

# MODEL_ID can point at a local folder to try a model before publishing it.
MODEL_ID = os.environ.get("MODEL_ID", "diegomponte/wav2vec2-base-speech-commands-35")
classifier = pipeline("audio-classification", model=MODEL_ID)


def classify(path: str | None) -> dict[str, float]:
    if path is None:
        return {}
    wave, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    clip = loudest_window(wave)
    predictions = classifier({"raw": clip, "sampling_rate": SAMPLE_RATE}, top_k=5)
    return {p["label"]: p["score"] for p in predictions}


WORDS = (
    "yes, no, up, down, left, right, on, off, stop, go, forward, backward, follow, learn, "
    "zero to nine, bed, bird, cat, dog, happy, house, marvin, sheila, tree, visual, wow"
)

demo = gr.Interface(
    fn=classify,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Say one word"),
    outputs=gr.Label(num_top_classes=5, label="Top 5"),
    title="Speech command recognition with Wav2Vec2",
    description=(
        "Say one English word from the list and the model guesses it. "
        f"Words it knows: {WORDS}. Only the loudest second of the recording is used.\n\n"
        f"Model: [{MODEL_ID}](https://huggingface.co/{MODEL_ID})"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
