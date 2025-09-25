"""Post-training analysis helpers for KLUE YNAT predictions."""
import argparse
import os
from pathlib import Path

import matplotlib


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze prediction CSV outputs")
    parser.add_argument("--pred_csv", type=str, default="outputs/predictions.csv")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--top_k", type=int, default=10, help="Number of low-confidence samples to save")
    parser.add_argument("--sample_incorrect", type=int, default=20, help="Number of incorrect samples to export")
    return parser.parse_args()


def main():
    args = parse_args()
    matplotlib.use("Agg")

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )

    pred_path = Path(args.pred_csv)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(pred_path)

    accuracy = accuracy_score(df["label"], df["prediction"])
    macro_f1 = f1_score(df["label"], df["prediction"], average="macro")

    label_report = classification_report(
        df["label"],
        df["prediction"],
        target_names=df["label_name"].unique(),
        digits=4,
    )

    label_accuracy = (
        df.groupby("label_name")["correct"].mean().sort_values(ascending=False)
    )

    report_txt = Path(args.out_dir) / "analysis_report.txt"
    with report_txt.open("w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write("Label-wise classification report\n")
        f.write(label_report)
        f.write("\n\nLabel-wise accuracy\n")
        f.write(label_accuracy.to_string())
    print(f"Saved analysis summary to {report_txt}")

    cm = confusion_matrix(df["label"], df["prediction"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = Path(args.out_dir) / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix heatmap to {cm_path}")

    low_conf = df.nsmallest(args.top_k, "probability")[
        ["text", "label_name", "prediction_name", "probability"]
    ]
    low_conf_path = Path(args.out_dir) / "lowest_confidence_samples.csv"
    low_conf.to_csv(low_conf_path, index=False)
    print(f"Saved low-confidence samples to {low_conf_path}")

    incorrect = df[df["correct"] == False].head(args.sample_incorrect)
    incorrect_path = Path(args.out_dir) / "incorrect_samples_head.csv"
    incorrect.to_csv(incorrect_path, index=False)
    print(f"Saved sample incorrect predictions to {incorrect_path}")


if __name__ == "__main__":
    main()
