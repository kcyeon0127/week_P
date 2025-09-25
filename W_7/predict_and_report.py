
"""Load a trained checkpoint, run predictions, and save evaluation summary."""
from pathlib import Path
from typing import List

import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from datamodule import YNATDataModule, YNAT_LABELS
from models import LightningBertClassifier


def main():
    # 기본 설정 값 (필요 시 수정)
    checkpoint_path = Path("outputs/bert-ynat-epoch=02-val_f1=0.000.ckpt")
    output_csv = Path("outputs/predictions.csv")
    model_name = "bert-base-multilingual-cased"
    batch_size = 32
    max_length = 128
    num_workers = 4
    split = "test"  # "val" 또는 "test"
    seed = 42

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightningBertClassifier.load_from_checkpoint(
        checkpoint_path,
        model_name=model_name,
    )
    model.to(device)
    model.eval()

    datamodule = YNATDataModule(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        seed=seed,
    )
    datamodule.setup()

    loader = datamodule.val_dataloader() if split == "val" else datamodule.test_dataloader()

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
    df.to_csv(output_csv, index=False)

    acc = accuracy_score(df["label"], df["prediction"])
    f1 = f1_score(df["label"], df["prediction"], average="macro")

    print(f"Saved predictions to {output_csv}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")


if __name__ == "__main__":
    main()
