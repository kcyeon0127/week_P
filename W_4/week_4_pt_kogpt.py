import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text_with_kogpt2(prompt_text, max_length=128):
    """
    사전 학습된 KoGPT2 모델을 사용하여 텍스트를 생성합니다.
    """
    # --- 1. 사전 학습된 KoGPT2 토크나이저와 모델 로드 ---
    model_name = "skt/kogpt2-base-v2"
    # 지정된 모델 이름으로 토크나이저를 불러옵니다.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 인과적 언어 모델링(Causal LM)을 위한 모델을 불러옵니다. (다음에 올 단어를 예측하는 모델)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # --- 2. 프롬프트 텍스트 인코딩 ---
    # 입력된 프롬프트 텍스트를 토큰 ID로 변환합니다. PyTorch 텐서 형태로 반환됩니다.
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    # --- 3. 장치 설정 ---
    # GPU 사용이 가능하면 GPU를, 그렇지 않으면 CPU를 사용하도록 설정합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = input_ids.to(device)

    # --- 4. 텍스트 생성 ---
    # 기울기 계산을 비활성화하여 추론 속도를 높입니다.
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=max_length,              # 생성할 텍스트의 최대 길이
            num_return_sequences=1,             # 생성할 시퀀스의 수
            pad_token_id=tokenizer.eos_token_id, # 패딩 토큰 ID를 문장 끝(EOS) 토큰 ID로 설정하여 경고를 방지
            eos_token_id=tokenizer.eos_token_id, # 문장 끝을 나타내는 토큰 ID
            do_sample=True,                     # 샘플링 전략을 사용하여 더 창의적인 텍스트 생성
            top_k=50,                           # 확률이 높은 상위 50개 단어 중에서만 샘플링
            top_p=0.95,                         # 누적 확률이 95%가 되는 단어 집합에서만 샘플링 (nucleus sampling)
            temperature=0.9                     # 생성될 단어의 확률 분포를 조절. 낮을수록 예측 가능한 텍스트 생성
        )

    # --- 5. 생성된 텍스트 디코딩 ---
    # 생성된 토큰 ID들을 사람이 읽을 수 있는 텍스트로 변환합니다.
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text

# 이 스크립트가 직접 실행될 때만 아래 코드를 실행합니다.
if __name__ == "__main__":
    # 텍스트 생성을 시작할 프롬프트를 정의합니다.
    prompt = "오늘 날씨가 좋아서"

    print(f"--- KoGPT2 텍스트 생성 ---")
    print(f"프롬프트: \"{prompt}\"")

    # 정의된 함수를 호출하여 텍스트를 생성합니다.
    result_text = generate_text_with_kogpt2(prompt)

    print("\n--- 생성된 텍스트 ---")
    print(result_text)
