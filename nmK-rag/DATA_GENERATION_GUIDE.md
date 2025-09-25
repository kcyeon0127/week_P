# 🤖 무료 API를 활용한 학습 데이터 자동 생성 가이드

## 개요
크롤링된 박물관 데이터를 기반으로 무료 API들을 활용하여 일반용/어린이용 응답 학습 데이터를 자동 생성하는 시스템입니다.

## 🎯 주요 기능

### ✅ **다양한 무료 API 지원**
- **Ollama** (로컬 무료, 추천) - llama3.2:3b, gemma2:2b 등
- **Groq** (클라우드 무료) - llama-3.1-8b-instant
- **HuggingFace** (무료 Inference API) - DialoGPT, GPT-Neo 등
- **OpenRouter** (무료 크레딧 $1) - phi-3-mini, zephyr-7b 등

### ✅ **크롤링 데이터 기반 생성**
- `data_curated/` 폴더의 박물관 소장품/전시 정보 활용
- 컨텐츠 타입 자동 분류 (소장품/전시/관람정보)
- 컨텐츠별 맞춤형 질문 템플릿 적용

### ✅ **멀티타겟 응답 생성**
- **일반용**: 전문적이고 상세한 설명 (200-300자)
- **어린이용**: 쉽고 재미있는 설명 + 이모지 (100-150자)

## 🚀 빠른 시작

### 1. 가장 쉬운 방법: Ollama 사용
```bash
# 1. Ollama 설치 및 실행
brew install ollama  # macOS
ollama serve

# 2. 모델 다운로드 (3GB, 처음만)
ollama pull llama3.2:3b

# 3. 학습 데이터 생성 (20개)
python generate_data.py --api ollama --count 20
```

### 2. 클라우드 API 사용
```bash
# Groq API (빠르고 무료)
export GROQ_API_KEY=gsk_your_key_here
python generate_data.py --api groq --count 30

# HuggingFace API
export HF_TOKEN=hf_your_token_here
python generate_data.py --api huggingface --count 15
```

## 📋 API 설정 가이드

### 🥇 **Ollama (추천)**
```bash
# 설치
brew install ollama          # macOS
curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# 모델 다운로드 옵션들
ollama pull llama3.2:3b      # 3GB, 빠름, 추천
ollama pull llama3.1:8b      # 8GB, 고품질
ollama pull gemma2:2b        # 2GB, 매우 빠름

# 실행
ollama serve
```

### 🥈 **Groq API (클라우드 무료)**
1. https://console.groq.com/ 회원가입
2. API Keys → Create API Key
3. `export GROQ_API_KEY=gsk_...`
4. **제한**: 시간당 14,400토큰, 분당 30요청

### 🥉 **HuggingFace**
1. https://huggingface.co/ 회원가입
2. Settings → Access Tokens → New token
3. `export HF_TOKEN=hf_...`
4. **제한**: 모델별 상이, 때때로 대기열 발생

### 🏅 **OpenRouter**
1. https://openrouter.ai/ 회원가입
2. Keys 탭에서 API key 생성
3. `export OPENROUTER_API_KEY=sk-or-...`
4. **크레딧**: $1 무료 제공 (약 1000번 호출)

## 💻 사용법

### 기본 데이터 생성
```bash
# API 설정 확인
python generate_data.py --check-setup

# 기본 생성 (Ollama, 30개)
python generate_data.py

# 다른 API 사용
python generate_data.py --api groq --count 50
python generate_data.py --api huggingface --count 20
```

### 데이터 병합 및 관리
```bash
# 기존 데이터와 병합
python merge_training_data.py merge \
  --original data/training_data.json \
  --generated data/generated_training_data.json \
  --output data/merged_training_data.json

# 데이터 분석
python merge_training_data.py analyze data/merged_training_data.json

# 훈련/검증 데이터 분할
python merge_training_data.py split data/merged_training_data.json --ratio 0.8
```

### 업데이트된 모델 훈련
```bash
# 병합된 데이터로 모델 재훈련
python src/fine_tuning.py --data_path data/merged_training_data.json
```

## 📊 생성 예시

### 입력 (크롤링된 소장품 데이터):
```json
{
  "title": "국립중앙박물관>소장품>소장품 검색",
  "text": "고죠다리의 요시쓰네와 벤케이의 싸움(우키요에)\n국적/시대: 일본 - 근대\n재질: 종이\n작가: 쓰키오카 요시토시...",
  "url": "https://www.museum.go.kr/MUSEUM/contents/M0502000000.do?relicId=36548568"
}
```

### 출력 (생성된 학습 데이터):
```json
{
  "question": "고죠다리의 요시쓰네와 벤케이 우키요에에 대해 알려줘",
  "general_response": "고죠다리의 요시쓰네와 벤케이의 싸움은 일본 근대 시대 쓰키오카 요시토시가 그린 우키요에 작품입니다. 이 작품은 일본의 유명한 전설인 미나모토노 요시쓰네와 무사승 벤케이의 운명적 만남을 묘사한 것으로...",
  "children_response": "이 그림은 일본의 유명한 이야기를 그린 예쁜 그림이에요! 강한 무사 두 명이 다리에서 싸우는 모습을 그렸는데, 나중에는 친구가 된다는 재미있는 이야기랍니다 ⚔️✨"
}
```

## ⚙️ 고급 설정

### 데이터 생성 파라미터 조정
`src/data_generator.py`에서 수정 가능:
```python
# 질문 템플릿 추가/수정
question_templates = {
    "소장품": [
        "새로운 질문 템플릿 추가...",
    ]
}

# 응답 길이 조정
general_prompt = "200-300자 내외"  # → "300-500자 내외"
children_prompt = "100-150자 내외"  # → "50-100자 내외"
```

### API 모델 변경
```python
# Ollama 모델 변경
export OLLAMA_MODEL=gemma2:9b

# OpenRouter 모델 선택
free_models = [
    "microsoft/phi-3-mini-128k-instruct:free",
    "새로운_무료_모델..."
]
```

## 📈 성능 최적화

### 생성 속도 향상
1. **Ollama**: 더 작은 모델 사용 (`gemma2:2b`)
2. **Groq**: 가장 빠름 (클라우드 GPU)
3. **배치 처리**: `--count` 적게 여러 번 실행
4. **API 호출 간격**: 코드에서 `time.sleep()` 조정

### 데이터 품질 향상
1. **프롬프트 개선**: 더 구체적인 지시사항 추가
2. **후처리**: 생성된 응답 검토 및 수동 수정
3. **필터링**: 부적절한 응답 자동 제거 로직 추가

## 🔍 트러블슈팅

### Ollama 연결 실패
```bash
# 서비스 확인
ollama serve

# 모델 확인
ollama list

# 포트 변경
export OLLAMA_URL=http://localhost:11435
```

### API 할당량 초과
- **Groq**: 시간당 제한, 잠시 후 재시도
- **HuggingFace**: 모델 대기열, 다른 모델 시도
- **OpenRouter**: 크레딧 소진 시 유료 충전 또는 다른 API 사용

### 생성 데이터 품질 문제
1. 프롬프트 개선: `src/data_generator.py`의 `create_response_prompts()` 수정
2. 모델 변경: 더 큰 모델 사용 (llama3.1:8b, gpt-4 등)
3. 후처리 로직 추가: 답변 길이, 언어, 관련성 검증

## 🎉 완료 후 다음 단계

1. **데이터 검증**: 생성된 데이터 품질 확인
2. **모델 재훈련**: 업데이트된 데이터로 LoRA 모델 재생성
3. **성능 평가**: 새로운 모델의 응답 품질 테스트
4. **배포**: Streamlit 앱에서 새 모델 사용

이제 무료 API를 활용하여 무제한으로 학습 데이터를 생성할 수 있습니다! 🚀