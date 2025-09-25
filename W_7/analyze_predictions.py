"""Post-training analysis helpers for KLUE YNAT predictions.

이 스크립트는 기본 경로(`outputs/predictions.csv`)를 기준으로 동작하며,
라벨별 성능 요약과 혼동행렬/저확신 샘플을 저장합니다.
"""

from pathlib import Path

import matplotlib


def main():
    # 기본 설정 (필요하면 아래 상수만 수정하면 됨)
    pred_csv = Path("outputs/predictions.csv")
    out_dir = Path("outputs")
    top_k = 10
    sample_incorrect = 20

    matplotlib.use("Agg")  # 서버 환경에서 안전하게 저장

    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )

    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(pred_csv)

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

    report_txt = out_dir / "analysis_report.txt"
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
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix heatmap to {cm_path}")

    low_conf = df.nsmallest(top_k, "probability")[
        ["text", "label_name", "prediction_name", "probability"]
    ]
    low_conf_path = out_dir / "lowest_confidence_samples.csv"
    low_conf.to_csv(low_conf_path, index=False)
    print(f"Saved low-confidence samples to {low_conf_path}")

    incorrect = df[df["correct"] == False].head(sample_incorrect)
    incorrect_path = out_dir / "incorrect_samples_head.csv"
    incorrect.to_csv(incorrect_path, index=False)
    print(f"Saved sample incorrect predictions to {incorrect_path}")


if __name__ == "__main__":
    main()
