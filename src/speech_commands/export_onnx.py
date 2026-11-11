"""Export the model to ONNX so it can run in the browser with Transformers.js.

    uv run --group export python -m speech_commands.export_onnx --model runs/wav2vec2/model

Writes `onnx/model.onnx` (fp32) and `onnx/model_quantized.onnx` (int8) inside the model folder,
the layout Transformers.js looks for.
"""

import argparse
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForAudioClassification

from speech_commands.data import CLIP_SAMPLES


class LogitsOnly(torch.nn.Module):
    """ONNX graphs return tensors, not ModelOutput objects."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        return self.model(input_values).logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    model = AutoModelForAudioClassification.from_pretrained(str(args.model)).eval()
    out = args.model / "onnx"
    out.mkdir(exist_ok=True)
    fp32 = out / "model.onnx"
    torch.onnx.export(
        LogitsOnly(model),
        (torch.zeros(1, CLIP_SAMPLES),),
        str(fp32),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={"input_values": {0: "batch", 1: "samples"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    # int8 weights for the transformer's matrix multiplications only: a 3x smaller download for
    # the browser at almost no accuracy cost. Quantizing the convolutional feature encoder too
    # makes the file 4x smaller but drops test accuracy from 97.7% to 78.4% (measured).
    quantize_dynamic(
        str(fp32),
        str(out / "model_quantized.onnx"),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )
    for f in sorted(out.iterdir()):
        print(f"{f.name}: {f.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
