"""Evaluate a saved model on the official test split and write the numbers the model card uses.

    uv run python -m speech_commands.evaluate --data ~/datasets/speech_commands \
        --model runs/wav2vec2/model --out results/test_metrics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from speech_commands.data import SpeechCommandsDataset, split_files
from speech_commands.metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="folder or Hub repo id")
    parser.add_argument("--out", type=Path, default=Path("results/test_metrics.json"))
    parser.add_argument("--split", default="testing", choices=["validation", "testing"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForAudioClassification.from_pretrained(str(args.model)).to(device).eval()
    extractor = AutoFeatureExtractor.from_pretrained(str(args.model))
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    files = split_files(args.data)[args.split]
    loader = DataLoader(
        SpeechCommandsDataset(args.data, files, labels, extractor), batch_size=64, num_workers=4
    )

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            all_logits.append(model(batch["input_values"].to(device)).logits.float().cpu().numpy())
            all_labels.append(batch["labels"].numpy())
    logits, y_true = np.concatenate(all_logits), np.concatenate(all_labels)
    y_pred = logits.argmax(-1)

    report = classification_report(
        y_true, y_pred, labels=range(len(labels)), target_names=labels, output_dict=True
    )
    results = {
        "split": args.split,
        "samples": int(len(y_true)),
        **compute_metrics((logits, y_true)),
        "per_class_f1": {label: round(report[label]["f1-score"], 4) for label in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "labels": labels,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"{args.split}: n={results['samples']} accuracy={results['accuracy']:.4f} "
        f"macro_f1={results['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
