# 🚀 국립중앙박물관 멀티타겟 RAG 챗봇 - 서버 실행 가이드

## 📋 **진행해야 할 단계**

### 1️⃣ **Ollama 설치 (1번만 실행)**

```bash
# 서버에서 Ollama 설치
curl -fsSL https://ollama.ai/install.sh | sh

# Ollama 서비스 시작 (백그라운드)
ollama serve &

# 설치 확인
ollama --version
curl http://localhost:11434/api/tags
```

### 2️⃣ **아나콘다 가상환경 생성 (1번만 실행)**

```bash
# Python 3.11 가상환경 생성
conda create -n nmk-rag python=3.11 -y

# 가상환경 활성화
conda activate nmk-rag

# 프로젝트 폴더로 이동
cd /path/to/nmK-rag

# Python 패키지 설치
pip install -r requirements.txt
```

### 3️⃣ **AI 모델 다운로드**

```bash
# 가상환경 활성화된 상태에서

# 추천 모델 (2GB, 중간 성능)
ollama pull llama3.2:3b

# 또는 가벼운 모델 (1.6GB, 빠름)
# ollama pull gemma2:2b

# 또는 고성능 모델 (4.7GB, GPU 서버 권장)
# ollama pull llama3.1:8b

# 다운로드 확인
ollama list
```

### 4️⃣ **학습 데이터 생성**

```bash
# 가상환경 활성화 (매번 필요)
conda activate nmk-rag

# 자동 데이터 생성 실행
python server_generate.py
```

실행하면 다음과 같이 나옵니다:
```
🐍 아나콘다 환경 감지됨: nmk-rag
🖥️  서버 리소스 체크:
   💾 총 메모리: 32.0GB
   🚀 CPU 코어: 16개
   🎮 GPU: ✅ RTX 4090 감지됨

⚙️ 최적화 설정:
   🎮 GPU 서버 고성능 모드 활성화!
   🤖 모델: llama3.2:3b
   📦 배치 크기: 15개
   🔄 병렬 처리: 4개

생성할 데이터 개수를 입력하세요 (기본: 30):
```

**원하는 개수 입력하고 엔터** (예: 50)

### 5️⃣ **완료까지 대기 (자동 진행)**

- 배치별로 자동 생성됨 (15개씩)
- 진행률 실시간 표시
- 완료까지 약 30-60분 소요

### 6️⃣ **생성된 데이터 확인**

```bash
# 생성된 파일들 확인
ls -la generated_data/

# 데이터 품질 분석
python merge_training_data.py analyze generated_data/all_generated_*.json

# 기존 데이터와 병합
python merge_training_data.py merge \
  --original data/training_data.json \
  --generated generated_data/all_generated_*.json \
  --output data/final_training_data.json
```

### 7️⃣ **모델 재훈련**

```bash
# 새로운 데이터로 LoRA 모델 훈련
python src/fine_tuning.py

# 완료까지 약 30분-2시간 소요
```

### 8️⃣ **Streamlit 앱 테스트**

```bash
# 웹 앱 실행
streamlit run app.py

# 브라우저에서 확인
# http://서버IP:8501
```

---

## ⚡ **빠른 실행 (한 번에 복사)**

```bash
# 1. Ollama 설치 & 실행
curl -fsSL https://ollama.ai/install.sh | sh && ollama serve &

# 2. 가상환경 생성 & 활성화
conda create -n nmk-rag python=3.11 -y && conda activate nmk-rag

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 모델 다운로드
ollama pull llama3.2:3b

# 5. 데이터 생성
python server_generate.py
```

---

## 🔧 **문제 해결**

### Ollama 연결 안 됨
```bash
# Ollama 프로세스 확인
ps aux | grep ollama

# 재시작
pkill ollama && ollama serve &

# 연결 테스트
curl http://localhost:11434/api/tags
```

### 가상환경 찾을 수 없음
```bash
# 가상환경 목록 확인
conda env list

# 다시 활성화
conda activate nmk-rag
```

### GPU 인식 안 됨
```bash
# GPU 상태 확인
nvidia-smi

# CUDA 버전 확인
nvcc --version
```

### 메모리 부족
```bash
# 더 작은 모델 사용
ollama pull gemma2:2b

# 환경변수 설정
export OLLAMA_MODEL=gemma2:2b
```

---

## 📊 **예상 성능**

| 서버 사양 | 추천 모델 | 50개 생성 시간 | 품질 |
|-----------|-----------|----------------|------|
| 16GB RAM | llama3.2:3b | 45-60분 | ⭐⭐⭐⭐ |
| 32GB RAM + GPU | llama3.1:8b | 20-30분 | ⭐⭐⭐⭐⭐ |
| 8GB RAM | gemma2:2b | 60-90분 | ⭐⭐⭐ |

---

## 📁 **프로젝트 구조**

```
nmK-rag/
├── src/
│   ├── data_generator.py       # 데이터 생성 엔진
│   ├── improved_rag.py         # 개선된 RAG 시스템
│   ├── multitarget_llm.py     # 멀티타겟 LLM
│   └── fine_tuning.py         # LoRA 파인튜닝
├── data/
│   └── training_data.json     # 기존 학습 데이터
├── generated_data/            # 생성된 데이터 (자동 생성됨)
├── server_generate.py         # 🎯 메인 실행 파일
├── generate_data.py          # 간단 실행 파일
├── merge_training_data.py    # 데이터 병합 도구
└── app.py                    # Streamlit 웹앱
```

---

## 🎯 **체크리스트**

- [ ] Ollama 설치 완료
- [ ] 가상환경 생성 완료
- [ ] 패키지 설치 완료
- [ ] 모델 다운로드 완료
- [ ] 데이터 생성 실행
- [ ] 결과 파일 확인
- [ ] 데이터 병합 완료
- [ ] 모델 재훈련 완료
- [ ] 웹앱 테스트 완료

**모든 것이 준비되었습니다! 순서대로 실행하시면 됩니다!** 🚀