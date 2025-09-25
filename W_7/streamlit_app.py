"""Streamlit demo for KLUE YNAT topic classification."""
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer

from datamodule import YNAT_LABELS
from models import LightningBertClassifier
from utils_checkpoint import find_latest_checkpoint


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "predictions.csv"


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: Optional[str], model_name: str):
    """Load Lightning module and tokenizer for inference."""
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightningBertClassifier.load_from_checkpoint(checkpoint_path, model_name=model_name)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer


def show_example_table(pred_csv: Path):
    st.header("예제 데이터 결과")
    if pred_csv.exists():
        df = pd.read_csv(pred_csv)
        display_df = df[["text", "label_name", "prediction_name", "correct", "probability"]].head(5)
        display_df = display_df.rename(
            columns={
                "text": "문장",
                "label_name": "정답",
                "prediction_name": "예측",
                "correct": "정답 여부",
                "probability": "확신도",
            }
        )
        st.dataframe(display_df)
    else:
        st.info("predictions.csv 파일을 찾을 수 없습니다. 평가 스크립트를 먼저 실행하세요.")


def predict_single(model: LightningBertClassifier, tokenizer: AutoTokenizer, text: str):
    device = next(model.parameters()).device
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs["logits"], dim=-1)
    probs = probs.squeeze(0).cpu()
    top_idx = int(probs.argmax().item())
    return top_idx, float(probs[top_idx].item()), probs


def show_live_demo(model: Optional[LightningBertClassifier], tokenizer: Optional[AutoTokenizer]):
    st.header("실시간 예측기")
    if model is None or tokenizer is None:
        st.error("모델 체크포인트를 불러오지 못했습니다. 학습을 먼저 실행하세요.")
        return

    user_text = st.text_area("분류할 문장을 입력하세요", "한국 경제가 빠르게 회복되고 있다.")
    if st.button("예측 실행"):
        if not user_text.strip():
            st.warning("문장을 입력해주세요.")
            return
        idx, conf, probs = predict_single(model, tokenizer, user_text)
        result_df = pd.DataFrame(
            {
                "예측 라벨": [YNAT_LABELS[idx]],
                "확신도": [conf],
            }
        )
        st.subheader("예측 결과")
        st.table(result_df)

        prob_df = pd.DataFrame({"라벨": YNAT_LABELS, "확률": probs.tolist()})
        st.bar_chart(prob_df.set_index("라벨"))


def main():
    st.set_page_config(page_title="KLUE YNAT 분류 데모", layout="wide")
    st.title("BERT 기반 뉴스 토픽 분류 데모")

    checkpoint_dir = BASE_DIR / "outputs"
    latest_ckpt = find_latest_checkpoint(str(checkpoint_dir))
    model, tokenizer = load_model(latest_ckpt, "bert-base-multilingual-cased")

    show_example_table(DEFAULT_OUTPUT)
    show_live_demo(model, tokenizer)


if __name__ == "__main__":
    main()
