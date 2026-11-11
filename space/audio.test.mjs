import assert from "node:assert/strict";
import { test } from "node:test";

import { loudestWindow, toMono } from "./audio.js";

test("short recordings are zero-padded to the window length", () => {
  const out = loudestWindow(new Float32Array([1, 1, 1]), 5);
  assert.deepEqual(Array.from(out), [1, 1, 1, 0, 0]);
});

test("long recordings keep the window with the most energy", () => {
  const wave = new Float32Array(30).fill(0.01);
  wave.fill(0.9, 12, 22);
  const out = loudestWindow(wave, 10);
  assert.equal(out.length, 10);
  assert.ok(out.every((v) => Math.abs(v - 0.9) < 1e-6));
});

test("a recording of exactly the window length is returned unchanged", () => {
  const wave = Float32Array.from({ length: 8 }, (_, i) => i);
  assert.deepEqual(Array.from(loudestWindow(wave, 8)), Array.from(wave));
});

test("stereo is averaged into mono", () => {
  const mono = toMono([new Float32Array([1, 0.5]), new Float32Array([0, 0.5])]);
  assert.deepEqual(Array.from(mono), [0.5, 0.5]);
});
