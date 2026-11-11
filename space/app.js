import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

import { loudestWindow, SAMPLE_RATE, toMono } from "./audio.js";

const MODEL_ID = "diegomponte/wav2vec2-base-speech-commands-35";
const RECORD_MS = 2000;

const status = document.getElementById("status");
const results = document.getElementById("results");
const player = document.getElementById("player");
const controls = document.querySelectorAll("button, input");

function setBusy(busy) {
  for (const el of controls) el.disabled = busy;
}

/** Decode any audio the browser understands and resample it to 16 kHz mono. */
async function decode(blob) {
  const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  try {
    const buffer = await ctx.decodeAudioData(await blob.arrayBuffer());
    const channels = Array.from({ length: buffer.numberOfChannels }, (_, i) =>
      buffer.getChannelData(i),
    );
    return toMono(channels);
  } finally {
    ctx.close();
  }
}

function show(predictions) {
  results.replaceChildren(
    ...predictions.map(({ label, score }) => {
      const li = document.createElement("li");
      const pct = (score * 100).toFixed(1);
      li.innerHTML = `<span class="label"></span><span class="bar"><span style="width:${pct}%"></span></span><span class="score">${pct}%</span>`;
      li.querySelector(".label").textContent = label;
      return li;
    }),
  );
}

let classifier;

async function classify(blob) {
  setBusy(true);
  status.textContent = "Listening…";
  try {
    player.src = URL.createObjectURL(blob);
    player.hidden = false;
    const clip = loudestWindow(await decode(blob));
    show(await classifier(clip, { top_k: 5 }));
    status.textContent = "Done. Try another word.";
  } catch (err) {
    console.error(err);
    status.textContent = `Could not read that audio: ${err.message}`;
  } finally {
    setBusy(false);
  }
}

async function record() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    status.textContent = "Microphone access was denied. You can upload a file instead.";
    return;
  }
  setBusy(true);
  const recorder = new MediaRecorder(stream);
  const chunks = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  const stopped = new Promise((resolve) => (recorder.onstop = resolve));
  recorder.start();
  status.textContent = "Recording… say one word now.";
  setTimeout(() => recorder.stop(), RECORD_MS);
  await stopped;
  for (const track of stream.getTracks()) track.stop();
  await classify(new Blob(chunks, { type: recorder.mimeType }));
}

document.getElementById("record").addEventListener("click", record);
document.getElementById("upload").addEventListener("change", (e) => {
  const [file] = e.target.files;
  if (file) classify(file);
});
for (const button of document.querySelectorAll("[data-example]")) {
  button.addEventListener("click", async () => {
    const response = await fetch(`examples/${button.dataset.example}.wav`);
    classify(await response.blob());
  });
}

try {
  classifier = await pipeline("audio-classification", MODEL_ID, { dtype: "q8" });
  status.textContent = "Ready. Record a word, upload a file or play an example.";
  setBusy(false);
} catch (err) {
  console.error(err);
  status.textContent = `The model failed to load: ${err.message}`;
}
