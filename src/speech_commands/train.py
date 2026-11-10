"""Fine-tune Wav2Vec2 on the full Speech Commands v0.02 training split.

uv run python -m speech_commands.train --data ~/datasets/speech_commands --out runs/wav2vec2
"""

import argparse
from pathlib import Path

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from speech_commands.data import SpeechCommandsDataset, list_labels, split_files
from speech_commands.metrics import compute_metrics, label_maps

BASE_MODEL = "facebook/wav2vec2-base"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="Speech Commands v0.02 folder")
    parser.add_argument("--out", type=Path, default=Path("runs/wav2vec2"))
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    labels = list_labels(args.data)
    splits = split_files(args.data)
    label2id, id2label = label_maps(labels)
    extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)
    train = SpeechCommandsDataset(args.data, splits["training"], labels, extractor)
    validation = SpeechCommandsDataset(args.data, splits["validation"], labels, extractor)

    model = AutoModelForAudioClassification.from_pretrained(
        BASE_MODEL, num_labels=len(labels), label2id=label2id, id2label=id2label
    )
    # The convolutional feature encoder already turns audio into good frame features;
    # only the transformer and the new head need to adapt to the 35 commands.
    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_steps=0.1,  # a float is a fraction of all steps in transformers 5
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100,
        dataloader_num_workers=4,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # Best checkpoint by validation accuracy; the test split stays untouched until evaluate.py.
    trainer.save_model(str(args.out / "model"))
    extractor.save_pretrained(str(args.out / "model"))
    print(trainer.evaluate())


if __name__ == "__main__":
    main()
