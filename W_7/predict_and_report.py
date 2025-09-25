"""Load a trained checkpoint, run predictions, and save evaluation summary."""
import argparse
import os
from typing import List

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from datamodule import YNATDataModule, YNAT_LABELS
from models import LightningBertClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BERT classifier and store predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--output_csv", type=str, default="outputs/predictions.csv")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    return parser.parse_args()


def run_prediction(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LightningBertClassifier.load_from_checkpoint(
        args.checkpoint,
        model_name=args.model_name,
    )
    model.to(device)
    model.eval()

    datamodule = YNATDataModule(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        seed=42,
    )
    datamodule.setup()

    loader = datamodule.val_dataloader() if args.split == "val" else datamodule.test_dataloader()

    rows: List[dict] = []
    for batch in loader:
        texts = batch["text"]
        labels = batch["labels"].numpy()
        inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k not in {"text", "labels"}
        }
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs["logits"], dim=-1)
            preds = probs.argmax(dim=-1).cpu().numpy()
        max_probs = probs.max(dim=-1).values.cpu().numpy()

        for text, label_idx, pred_idx, prob in zip(texts, labels, preds, max_probs):
            rows.append(
                {
                    "text": text,
                    "label": label_idx,
                    "label_name": YNAT_LABELS[label_idx],
                    "prediction": pred_idx,
                    "prediction_name": YNAT_LABELS[pred_idx],
                    "probability": float(prob),
                    "correct": bool(label_idx == pred_idx),
                }
            )

    df = pd.DataFrame(rows)
    return df


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    df = run_prediction(args)
    df.to_csv(args.output_csv, index=False)

    acc = accuracy_score(df["label"], df["prediction"])
    f1 = f1_score(df["label"], df["prediction"], average="macro")

    print(f"Saved predictions to {args.output_csv}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")


if __name__ == "__main__":
    main()
