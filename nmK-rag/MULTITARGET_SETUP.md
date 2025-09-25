# 멀티타겟 RAG 챗봇 설정 가이드

## 개요
이 시스템은 일반 관람객과 어린이 관람객을 위한 서로 다른 응답 스타일을 제공하는 멀티타겟 RAG 챗봇입니다.

- **일반 모드**: 전문적이고 상세한 설명
- **어린이 모드**: "신라 금관은 예쁜 왕관이에요. 아주 오래전에 신라 사람들이 썼던 거랍니다 🌟" 스타일

## 1. 환경 설정

### 1.1 의존성 설치
```bash
pip install -r requirements.txt
```

### 1.2 GPU 환경 확인
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
```

## 2. 모델 학습

### 2.1 학습 데이터 준비
학습 데이터는 이미 `data/training_data.json`에 준비되어 있습니다:
- 10개의 질문-답변 쌍
- 각 질문마다 일반용/어린이용 응답 포함

### 2.2 LoRA 파인튜닝 실행
```bash
cd /path/to/nmK-rag
python src/fine_tuning.py
```

이 과정은 다음을 수행합니다:
1. `models/lora-general/` - 일반 관람객용 모델 생성
2. `models/lora-children/` - 어린이 관람객용 모델 생성

### 2.3 예상 소요 시간
- GPU 환경: 각 모델당 약 10-20분
- CPU 환경: 각 모델당 약 1-2시간

## 3. Streamlit 앱 실행

### 3.1 앱 시작
```bash
streamlit run app.py
```

### 3.2 UI 사용법
1. **사이드바**에서 "응답 모드" 선택:
   - "일반 관람객": 전문적인 답변
   - "어린이 관람객": 친근하고 쉬운 답변

2. **질문 입력**: 박물관 관련 질문 작성

3. **응답 확인**: 선택한 모드에 따른 맞춤형 답변

## 4. 테스트 예시

### 4.1 테스트 질문들
```
- "신라 금관에 대해 알려줘"
- "청자에 대해 설명해줘"
- "조선시대 백자는 어떤 특징이 있나요?"
- "불국사 석가탑의 의미는?"
```

### 4.2 예상 응답 차이
**일반 모드**: "신라 금관은 5-6세기 경주 지역 고분에서 발견된 금제 관모로, 신라 왕실의 위엄을 보여주는 중요한 유물입니다..."

**어린이 모드**: "신라 금관은 예쁜 왕관이에요. 아주 오래전에 신라 사람들이 썼던 거랍니다 🌟"

## 5. 고급 설정

### 5.1 모델 경로 수정
`src/multitarget_llm.py`에서 모델 경로 변경:
```python
model_configs = {
    "general": "/custom/path/to/lora-general",
    "children": "/custom/path/to/lora-children"
}
```

### 5.2 Ollama 사용
환경변수 설정으로 Ollama 사용 가능:
```bash
export OLLAMA_MODEL=llama3.1:8b
export OLLAMA_URL=http://localhost:11434
```

### 5.3 베이스 모델 변경
`src/fine_tuning.py`와 `src/multitarget_llm.py`에서 베이스 모델 수정:
```python
base_model_name = "Qwen/Qwen2.5-7B-Instruct"  # 더 큰 모델 사용
```

## 6. 트러블슈팅

### 6.1 메모리 부족
- 베이스 모델을 더 작은 것으로 변경
- 배치 사이즈 줄이기: `per_device_train_batch_size=1`
- LoRA rank 줄이기: `r=8`

### 6.2 모델 로드 실패
- 모델 파일 경로 확인
- 베이스 모델로 자동 fallback됨

### 6.3 응답 품질 개선
1. **더 많은 학습 데이터 추가**: `data/training_data.json` 확장
2. **에포크 증가**: `num_train_epochs=5`
3. **LoRA 파라미터 튜닝**: `r=32, lora_alpha=64`

## 7. 데이터셋 확장

새로운 질문-답변 쌍 추가:
```json
{
  "question": "새로운 질문",
  "general_response": "전문적인 답변",
  "children_response": "어린이 친화적 답변 🌟"
}
```

## 8. 배포

### 8.1 Docker 사용
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### 8.2 클라우드 배포
- Streamlit Cloud
- Hugging Face Spaces
- Google Colab

이제 멀티타겟 응답이 가능한 RAG 챗봇이 완성되었습니다! 🎉