.PHONY: test lint train evaluate export-onnx

DATA ?= $(HOME)/datasets/speech_commands
RUN ?= runs/wav2vec2

test:
	uv run pytest -q
	node --test space/audio.test.mjs

lint:
	uv run ruff check .
	uv run ruff format --check .

train:
	uv run python -m speech_commands.train --data $(DATA) --out $(RUN)

evaluate:
	uv run python -m speech_commands.evaluate --data $(DATA) --model $(RUN)/model --out results/test_metrics.json

export-onnx:
	uv run --group export python -m speech_commands.export_onnx --model $(RUN)/model
