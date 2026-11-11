// Pure audio helpers, shared by the page and the Node tests.

export const SAMPLE_RATE = 16000;

/** Average all channels of an AudioBuffer-like list of Float32Arrays into one. */
export function toMono(channels) {
  if (channels.length === 1) return channels[0];
  const mono = new Float32Array(channels[0].length);
  for (const channel of channels) {
    for (let i = 0; i < mono.length; i++) mono[i] += channel[i] / channels.length;
  }
  return mono;
}

/**
 * The model was trained on one-second clips. Short recordings are zero-padded; for long ones,
 * keep the window with the most energy, which is where the spoken word is.
 */
export function loudestWindow(wave, length = SAMPLE_RATE) {
  if (wave.length <= length) {
    const out = new Float32Array(length);
    out.set(wave);
    return out;
  }
  let energy = 0;
  for (let i = 0; i < length; i++) energy += wave[i] * wave[i];
  let best = energy;
  let bestStart = 0;
  for (let start = 1; start + length <= wave.length; start++) {
    energy += wave[start + length - 1] ** 2 - wave[start - 1] ** 2;
    if (energy > best) {
      best = energy;
      bestStart = start;
    }
  }
  return wave.slice(bestStart, bestStart + length);
}
