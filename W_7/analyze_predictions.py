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
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
        balanced_accuracy_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )
    from datamodule import YNAT_LABELS

    # ---------------------------
    # Label language and font setup
    # ---------------------------
    # If True, use English labels to avoid font issues in PNGs.
    USE_ENGLISH_LABELS = True
    EN_YNAT_LABELS = [
        "IT_SCIENCE",
        "ECONOMY",
        "SOCIETY",
        "LIFE_CULTURE",
        "WORLD",
        "SPORTS",
        "POLITICS",
    ]

    # Optional: if you want Korean labels and no system font supports them,
    # set this to True to try downloading a Korean font automatically.
    ALLOW_FONT_DOWNLOAD = False  # keep False by default

    def ensure_korean_font():
        import matplotlib.font_manager as fm
        import tempfile, requests
        # Common Korean-capable fonts that might already be present.
        candidates = [
            "Malgun Gothic",  # Windows
            "AppleGothic",    # macOS
            "NanumGothic",
            "Noto Sans CJK KR",
            "Noto Sans KR",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        for name in candidates:
            if name in available:
                plt.rcParams["font.family"] = name
                return True
        # Try to download NotoSansKR if allowed
        if not ALLOW_FONT_DOWNLOAD:
            return False
        url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansKR-Regular.otf"
        try:
            with requests.get(url, timeout=20) as r:
                r.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".otf", delete=False) as tmp:
                    tmp.write(r.content)
                    font_path = tmp.name
            fm.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = "Noto Sans KR"
            return True
        except Exception:
            return False

    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(pred_csv)

    y_true = df["label"].to_numpy()
    y_pred = df["prediction"].to_numpy()
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    macro_prec, macro_rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    selected_labels = EN_YNAT_LABELS if USE_ENGLISH_LABELS else YNAT_LABELS
    # Classification report with consistent class order 0..6
    label_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(selected_labels))),
        target_names=selected_labels,
        digits=4,
        zero_division=0,
    )

    # Per-class accuracy (recall per true class)
    per_class_acc = {}
    for i, name in enumerate(selected_labels):
        mask = (y_true == i)
        if mask.sum() == 0:
            per_class_acc[name] = float("nan")
        else:
            per_class_acc[name] = float((y_pred[mask] == i).mean())
    import pandas as pd
    label_accuracy = pd.Series(per_class_acc).sort_values(ascending=False)

    report_txt = out_dir / "analysis_report.txt"
    with report_txt.open("w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"Macro Precision: {macro_prec:.4f}, Macro Recall: {macro_rec:.4f}\n\n")
        f.write("Label-wise classification report\n")
        f.write(label_report)
        f.write("\n\nLabel-wise accuracy\n")
        f.write(label_accuracy.to_string())
    print(f"Saved analysis summary to {report_txt}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    # If using Korean labels and font is unavailable, try enabling a Korean font.
    if not USE_ENGLISH_LABELS:
        ensure_korean_font()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=selected_labels, yticklabels=selected_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix heatmap to {cm_path}")

    # Row-normalized confusion matrix (recall perspective)
    with np.errstate(invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0,
                xticklabels=selected_labels, yticklabels=selected_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True (row-normalized)")
    plt.tight_layout()
    cmn_path = out_dir / "confusion_matrix_normalized.png"
    plt.savefig(cmn_path, dpi=200)
    plt.close()
    print(f"Saved normalized confusion matrix heatmap to {cmn_path}")

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
