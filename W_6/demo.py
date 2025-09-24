
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from utils_checkpoint import find_latest_checkpoint

# --------------------------
# Utilities from week6_gpt2_stsb.py
# --------------------------
PROMPT_PREFIX = "Source: "
TARGET_PREFIX = "\nTarget: "
EOS = ""  # GPT-2 eos token

def build_prompt(s1: str) -> str:
    """추론 시 조건 프롬프트"""
    return f"{PROMPT_PREFIX}{s1}{TARGET_PREFIX}"

# --------------------------
# Streamlit App
# --------------------------
BASE_DIR = "runs/gpt2-stsb"
MODEL_DIR = find_latest_checkpoint(BASE_DIR)
EVAL_CSV_PATH = os.path.join(BASE_DIR, "eval_validation.csv")

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer."""
    if not os.path.exists(MODEL_DIR):
        return None, None
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    dtype = torch.float16 if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer

def generate_prediction(model, tokenizer, text):
    """Generate a prediction for a single text input."""
    prompt = build_prompt(text)
    device = model.device
    
    enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        # outputs = model.generate(
        #     **enc,
        #     max_new_tokens=64,
        #     do_sample=True,
        #     top_k=50,
        #     top_p=0.95,
        #     temperature=0.8,
        #     num_return_sequences=1,
        #     pad_token_id=tokenizer.eos_token_id,
        #     eos_token_id=tokenizer.eos_token_id,
        # )
        outputs = model.generate(
            **enc,
            max_new_tokens=64,
            num_beams=5,  # Beam search 사용
            no_repeat_ngram_size=2, # 같은 구문 반복 방지
            early_stopping=True, # 문장이 끝나면 생성 중단
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    print(f"Raw generated text: {decoded[0]}")
    
    if decoded and decoded[0].startswith(prompt):
        return decoded[0][len(prompt):].strip()
    else:
        return "Failed to generate a valid prediction."

def main():
    st.set_page_config(page_title="GPT-2 STS-B Demo", layout="wide")
    st.title("GPT-2 기반 문장 생성 데모 (STS-B)")

    model, tokenizer = load_model_and_tokenizer()

    if model is None or tokenizer is None:
        st.error(f"훈련된 모델을 찾을 수 없습니다. '{MODEL_DIR}' 디렉토리에 모델이 있는지 확인하세요. "
                 f"먼저 `python week6_gpt2_stsb.py train`을 실행해야 할 수 있습니다.")
        return

    st.header("예제 데이터 성능")
    if os.path.exists(EVAL_CSV_PATH):
        df_eval = pd.read_csv(EVAL_CSV_PATH)
        st.dataframe(df_eval.head())
    else:
        st.warning(f"평가 결과 파일(`{EVAL_CSV_PATH}`)을 찾을 수 없습니다. "
                   f"예제 테이블을 표시하려면 먼저 `python week6_gpt2_stsb.py predict` 및 "
                   f"`python week6_gpt2_stsb.py evaluate`를 실행하세요.")

    st.header("실시간 예측")
    input_text = st.text_input("문장을 입력하세요:", "A man is playing a guitar.")

    if st.button("예측 생성"):
        if input_text:
            with st.spinner("생성 중..."):
                prediction = generate_prediction(model, tokenizer, input_text)
                
                st.subheader("결과")
                result_df = pd.DataFrame({
                    "입력 문장": [input_text],
                    "모델 예측": [prediction]
                })
                st.table(result_df)
                st.info("참고: 실시간 예측에서는 정답(reference) 문장이 없으므로, BLEU, ROUGE-L, BERTScore와 같은 성능 지표는 계산되지 않습니다.")
        else:
            st.warning("예측을 생성하려면 문장을 입력하세요.")

if __name__ == "__main__":
    main()
