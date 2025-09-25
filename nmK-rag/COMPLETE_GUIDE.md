# 🏛️ 국립중앙박물관 AI 도슨트 - 완전 가이드

## 📖 **프로젝트 개요**

국립중앙박물관 웹사이트 데이터를 기반으로 **멀티타겟 RAG 시스템**을 구축하여, 일반 관람객과 어린이 관람객을 위한 맞춤형 응답을 제공하는 AI 도슨트입니다.

### ✨ **핵심 기능**
- 🤖 **멀티타겟 응답**: 일반용(전문적) / 어린이용(친근함) 구분
- ⚡ **무료 API 기반**: Ollama, Groq, HuggingFace 등 활용
- 🔍 **지능형 검색**: 교통수단별 정확한 정보 제공
- ⏰ **실시간 맥락**: 현재/과거 전시 자동 구분
- 🎯 **자동 데이터 생성**: 크롤링된 데이터로 학습 데이터 무제한 생성

---

## 🚀 **빠른 시작 가이드**

### 📋 **실행 순서 (서버에서)**

#### **1단계: 기본 환경 설정**
```bash
# 1. Ollama 설치 (1분)
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &

# 2. 아나콘다 가상환경 생성 (2분)
conda create -n rag_bot python=3.11 -y
conda activate rag_bot

# 3. 프로젝트 이동 & 패키지 설치
cd /path/to/nmK-rag
pip install -r requirements.txt
```

#### **2단계: AI 모델 다운로드**
```bash
# 서버 사양에 맞는 모델 선택

# 추천: 중간 성능 (2GB)
ollama pull llama3.2:3b

# 또는 가벼운 모델 (1.6GB) - 저사양 서버
ollama pull gemma2:2b

# 또는 고성능 모델 (4.7GB) - GPU 서버
ollama pull llama3.1:8b

# 다운로드 확인
ollama list
```

#### **3단계: 학습 데이터 자동 생성**
```bash
# 가상환경 활성화 (매번 필요)
conda activate nmk-rag

# 자동 데이터 생성 실행
python server_generate.py
```

실행 화면:
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

생성할 데이터 개수를 입력하세요: 50
```

#### **4단계: 데이터 처리 및 모델 훈련**
```bash
# 생성 완료 후 데이터 병합
python merge_training_data.py merge \
  --original data/training_data.json \
  --generated generated_data/all_generated_*.json \
  --output data/final_training_data.json

# LoRA 모델 훈련
python src/fine_tuning.py

# Streamlit 앱 실행
streamlit run app.py
```

---

## 📊 **성능 및 예상 시간**

### **서버별 성능**
| 서버 사양 | 추천 모델 | 50개 생성 시간 | 품질 | 배치 크기 |
|-----------|-----------|----------------|------|-----------|
| **GPU 서버** (32GB+GPU) | llama3.1:8b | **20-30분** | ⭐⭐⭐⭐⭐ | 20개 |
| 중간 서버 (16GB RAM) | llama3.2:3b | 45-60분 | ⭐⭐⭐⭐ | 15개 |
| 저사양 서버 (8GB RAM) | gemma2:2b | 60-90분 | ⭐⭐⭐ | 5개 |

### **응답 품질 예시**

#### **입력 (크롤링된 데이터)**:
```json
{
  "title": "신라금관",
  "text": "신라 금관은 5-6세기 경주 지역 고분에서 발견된 금제 관모..."
}
```

#### **생성된 응답**:
- **일반용**: "신라 금관은 5-6세기 경주 지역 고분에서 발견된 금제 관모로, 신라 왕실의 위엄을 보여주는 중요한 유물입니다. 나무 모양의 세움 장식과 달개가 특징적이며..."

- **어린이용**: "신라 금관은 예쁜 왕관이에요. 아주 오래전에 신라 사람들이 썼던 거랍니다 🌟"

---

## 🎯 **주요 개선사항**

### **1. 멀티타겟 응답 시스템**
- **일반 관람객**: 전문적이고 상세한 설명 (200-300자)
- **어린이 관람객**: 쉽고 재미있는 설명 + 이모지 (100-150자)
- **Streamlit UI**: 사이드바에서 응답 모드 실시간 전환 가능

### **2. 지능형 교통 정보 검색**
- **문제 해결**: "지하철로 가는 방법" → 지하철 정보만 제공
- **의도 파악**: 질문에서 교통수단 자동 감지
- **점수 조정**: 관련 문서 우선순위 자동 조정

### **3. 무제한 데이터 생성**
- **무료 API 활용**: Ollama(로컬), Groq, HuggingFace 등
- **자동 품질 관리**: 응답 길이, 스타일, 내용 자동 검증
- **배치 처리**: 서버 리소스에 맞춰 자동 최적화

---

## 📁 **프로젝트 구조**

```
nmK-rag/
├── 🎯 핵심 실행 파일
│   ├── server_generate.py         # 메인 데이터 생성 스크립트
│   ├── app.py                     # Streamlit 멀티타겟 웹앱
│   └── merge_training_data.py     # 데이터 병합 도구
│
├── 🤖 AI 시스템
│   ├── src/improved_rag.py        # 개선된 RAG (교통정보 정확도 향상)
│   ├── src/multitarget_llm.py     # 멀티타겟 LLM 시스템
│   ├── src/data_generator.py      # 무료 API 데이터 생성 엔진
│   └── src/fine_tuning.py         # LoRA 파인튜닝
│
├── 📊 데이터
│   ├── data/training_data.json    # 기존 학습 데이터
│   ├── data_curated/              # 크롤링된 박물관 데이터
│   └── generated_data/            # 생성된 학습 데이터 (자동 생성)
│
└── 📚 문서
    ├── COMPLETE_GUIDE.md          # 📖 이 파일 (통합 가이드)
    ├── START_HERE.md              # 🚀 빠른 시작 가이드
    └── requirements.txt           # Python 의존성
```

---

## 🔧 **문제 해결 가이드**

### **Ollama 연결 실패**
```bash
# 프로세스 확인
ps aux | grep ollama

# 재시작
pkill ollama && ollama serve &

# 연결 테스트
curl http://localhost:11434/api/tags
```

### **메모리 부족**
```bash
# 더 작은 모델 사용
ollama pull gemma2:2b
export OLLAMA_MODEL=gemma2:2b

# 배치 크기 자동 조정됨 (server_generate.py)
```

### **GPU 인식 안 됨**
```bash
# GPU 상태 확인
nvidia-smi

# CUDA 설치 확인
nvcc --version

# Ollama GPU 사용 확인
ollama pull llama3.1:8b  # GPU 모델로 테스트
```

### **가상환경 문제**
```bash
# 가상환경 목록 확인
conda env list

# 재생성
conda create -n nmk-rag python=3.11 -y
conda activate nmk-rag
pip install -r requirements.txt
```

---

## ⚙️ **고급 설정 및 커스터마이징**

### **1. API 모델 변경**
```bash
# Ollama 모델 변경
export OLLAMA_MODEL=llama3.1:8b  # 고성능
export OLLAMA_MODEL=gemma2:2b    # 가벼움

# 다른 무료 API 사용
export GROQ_API_KEY=your_key
python generate_data.py --api groq --count 100
```

### **2. 응답 스타일 커스터마이징**
`src/data_generator.py` 수정:
```python
# 응답 길이 조정
general_prompt = "300-500자 내외로 더 상세히"
children_prompt = "50-100자로 더 간결하게"

# 질문 템플릿 추가
question_templates = {
    "소장품": [
        "이 유물의 제작 기법은?",
        "이 작품의 문화적 의미는?",
        # 새 템플릿 추가...
    ]
}
```

### **3. 검색 정확도 개선**
`src/improved_rag.py` 수정:
```python
# 교통수단별 키워드 추가
subway_keywords = ["지하철", "전철", "역", "호선", "새로운_키워드"]

# 점수 가중치 조정
if transport_intent == "subway":
    relevance_score *= 3.0  # 지하철 관련성 더 높임
```

---

## 🎉 **완료 체크리스트**

### **기본 설치**
- [ ] Ollama 설치 및 실행 확인
- [ ] 아나콘다 가상환경 생성
- [ ] Python 패키지 설치 완료
- [ ] AI 모델 다운로드 완료

### **데이터 생성**
- [ ] `python server_generate.py` 실행
- [ ] 원하는 개수 입력 (예: 50개)
- [ ] 생성 완료까지 대기 (30-90분)
- [ ] `generated_data/` 폴더에 결과 파일 확인

### **모델 훈련**
- [ ] 데이터 병합: `merge_training_data.py merge`
- [ ] 데이터 분석: `merge_training_data.py analyze`
- [ ] LoRA 훈련: `python src/fine_tuning.py`
- [ ] 훈련 완료 확인 (30분-2시간)

### **테스트 및 배포**
- [ ] Streamlit 앱 실행: `streamlit run app.py`
- [ ] 브라우저에서 접속: `http://서버IP:8501`
- [ ] 응답 모드 전환 테스트 (일반 ↔ 어린이)
- [ ] 교통 정보 정확도 테스트 ("지하철로 가는 방법")

---

## 🌟 **프로젝트 특장점**

### **✅ 완전 무료 시스템**
- API 비용 없음 (Ollama 로컬 실행)
- 무제한 데이터 생성 가능
- 오픈소스 모델 활용

### **✅ 실용성**
- 실제 박물관 데이터 기반
- 관람객 질문 패턴 반영
- 현실적인 응답 품질

### **✅ 확장성**
- 새로운 API 쉽게 추가 가능
- 다른 도메인으로 확장 가능
- 모델 교체 및 업그레이드 용이

### **✅ 사용자 친화성**
- 원클릭 실행 스크립트
- 자동 리소스 최적화
- 실시간 진행상황 표시

---

## 🎯 **다음 단계 제안**

### **단기 개선사항**
1. **데이터 품질 향상**: 더 많은 학습 데이터 생성 (100-500개)
2. **응답 다양성**: 추가 응답 스타일 개발 (청소년용, 전문가용)
3. **정확도 개선**: 더 큰 모델 활용 (llama3.1:8b, qwen2.5:7b)

### **장기 확장**
1. **음성 인터페이스**: STT/TTS 연동
2. **이미지 검색**: 유물 이미지 기반 검색
3. **다국어 지원**: 영어, 중국어, 일본어 응답
4. **실시간 업데이트**: 웹사이트 변경사항 자동 반영

---

**🏛️ 국립중앙박물관 AI 도슨트 시스템이 완성되었습니다!**
**모든 것이 준비되어 있으니 순서대로 실행하시면 됩니다! 🚀**