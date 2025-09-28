# 🏛️ 국립중앙박물관 AI 도슨트 - 멀티타겟 RAG 시스템

국립중앙박물관 웹사이트의 **2194개 실제 데이터**를 기반으로 구축된 **멀티타겟 RAG(Retrieval Augmented Generation) 시스템**입니다. 일반 관람객과 어린이 관람객을 위한 **차별화된 응답 스타일**을 제공하는 AI 도슨트 챗봇입니다.

## ✨ 핵심 기능

### 🎯 **멀티타겟 응답 시스템**
- **일반용**: 전문적이고 상세한 박물관 해설 (평균 457.3자)
- **어린이용**: 친근하고 재미있는 설명 + 이모지 (평균 334.6자)
- **실시간 모드 전환**: Streamlit 인터페이스에서 즉시 변경 가능

### 🤖 **자동 학습 데이터 생성**
- **무료 API 활용**: Ollama(Tesla V100 GPU), Groq, HuggingFace 등
- **무제한 확장**: 2194개 크롤링 데이터에서 자동으로 질문-답변 쌍 생성
- **품질 관리**: 중복 제거 및 응답 길이/스타일 자동 검증
- **실제 성과**: 2000개 생성 → 539개 고품질 데이터 선별

### 🔍 **지능형 검색 시스템**
- **교통수단별 정확한 정보**: "지하철로 가는 방법" → 지하철 정보만 제공
- **의도 파악**: 질문에서 교통수단 자동 감지
- **관련도 점수 조정**: 사용자 의도에 맞는 문서 우선순위 자동 조정

### 🚀 **LoRA 파인튜닝**
- **어린이 말투 학습**: 485개 훈련 + 54개 검증 데이터
- **효율적 학습**: 전체 모델의 1.18%만 파인튜닝
- **Tesla V100 최적화**: GPU 2,3번 활용으로 고속 처리

## 🏗️ 시스템 아키텍처

```
데이터 수집 → 정제/청킹 → 임베딩/인덱싱 → 하이브리드 검색 → LLM 생성 → AI 도슨트 UI
```

### 1️⃣ 데이터 수집 (Data Collection)
- **웹 크롤링**: 전시 정보, 해설 안내, 이용 안내
- **PDF 파싱**: 도록, 연구 자료 등
- **전시 상태 자동 분류**: 현재/예정/지난 전시 구분

### 2️⃣ 데이터 처리 (Data Processing)
- **텍스트 정제**: 노이즈 제거, 구조화
- **스마트 청킹**: 문단/문장 단위 분할 (400-1200자)
- **메타데이터 추출**: 전시 기간, 장소, 상태 등

### 3️⃣ 임베딩 & 인덱싱 (Embedding & Indexing)
- **Dense Vector**: BGE-M3 (다국어 임베딩)
- **Sparse Vector**: BM25 (키워드 매칭)
- **Vector DB**: ChromaDB (코사인 유사도)

### 4️⃣ 검색 & 생성 (Retrieval & Generation)
- **하이브리드 검색**: Dense + Sparse + CrossEncoder 리랭킹
- **시간 필터링**: 현재 정보 우선, 과거 정보 억제
- **LLM 생성**: Qwen2.5-1.5B-Instruct (인용 포함)

## 📁 디렉토리 구조

```
nmK-rag/
├── app.py                    # Streamlit AI 도슨트 웹 인터페이스
├── data_raw/                 # 크롤링된 원본 JSON 데이터
├── data_curated/             # 정제된 문서 및 청크 데이터
├── index/                    # ChromaDB 벡터 인덱스
├── src/                      # 핵심 로직 (리팩토링됨)
│   ├── crawl/                # 데이터 수집
│   │   ├── crawl.py          # 전시 정보 크롤러
│   │   ├── crawl_commentary.py    # 해설 안내 크롤러
│   │   ├── crawl_visitor_info.py  # 이용 안내 크롤러
│   │   └── crawl_utils.py    # 크롤링 유틸리티 (전시 상태 판단 등)
│   ├── constants.py          # 공통 상수 및 키워드 (신규)
│   ├── schema.py             # 데이터 스키마 (Doc, Chunk)
│   ├── curate_docs.py        # 텍스트 정제 및 노이즈 제거
│   ├── clean_chunk.py        # 문서 청킹 (400-1200자)
│   ├── embed_index.py        # BGE-M3 임베딩 & ChromaDB 인덱싱
│   ├── retriever.py          # 하이브리드 검색 (Dense+BM25+Rerank)
│   ├── multitarget_llm.py    # 멀티타겟 LLM (일반/어린이)
│   ├── rag.py                # 통합 RAG 시스템 (간소화됨)
│   ├── data_generator.py     # 훈련 데이터 자동 생성
│   ├── fine_tuning.py        # LoRA 파인튜닝
│   ├── eval.py               # 평가 메트릭 (Hit@k, MRR, Faithfulness)
│   └── parse_pdf.py          # PDF 문서 파싱
└── ai_docent_evaluations.csv # 사용자 평가 로그
```

## 🔄 전체 워크플로우

### Phase 1: 데이터 수집
```bash
# 1. 전시 정보 크롤링
cd src/crawl
python crawl.py

# 2. 해설 안내 크롤링
python crawl_commentary.py

# 3. 이용 안내 크롤링
python crawl_visitor_info.py

# 4. PDF 문서 파싱 (선택사항)
python src/parse_pdf.py "data_raw/**/*.pdf" -o data_curated
```

### Phase 2: 데이터 처리
```bash
# 1. 텍스트 정제 (노이즈 제거, 구조화)
python src/curate_docs.py data_raw data_curated

# 2. 문서 청킹 (400-1200자 단위)
python src/clean_chunk.py data_curated -o data_curated/chunks.jsonl
```

### Phase 3: 임베딩 & 인덱싱
```bash
# BGE-M3 임베딩 & ChromaDB 벡터 인덱스 구축
python src/embed_index.py \
    --chunks data_curated/chunks.jsonl \
    --persist index/chroma \
    --collection nmK \
    --model BAAI/bge-m3
```

### Phase 4: 멀티타겟 학습 데이터 생성
```bash
# Tesla V100 GPU로 대량 데이터 자동 생성
python server_generate.py
# 입력: 생성할 개수 (예: 1000개)
# 결과: 크롤링 데이터 → 멀티타겟 질문-답변 쌍

# 데이터 병합 및 품질 관리
python merge_training_data.py merge \
    --original data/training_data.json \
    --generated generated_data/all_generated_*.json \
    --output data/mega_training_data.json

# 데이터 분석 및 분할
python merge_training_data.py analyze data/mega_training_data.json
python merge_training_data.py split data/mega_training_data.json --ratio 0.9
```

### Phase 5: LoRA 파인튜닝
```bash
# 어린이 말투 LoRA 파인튜닝 (Tesla V100 최적화)
export CUDA_VISIBLE_DEVICES=2,3
python src/fine_tuning.py

# 결과: models/lora-general/, models/lora-children/
```

### Phase 6: AI 도슨트 실행
```bash
# Streamlit 멀티타겟 웹 인터페이스 실행
streamlit run app.py --server.port 8501
```

## 🧠 핵심 기술 상세

### 1. 지능형 전시 상태 판단 (`crawl_utils.py`)
```python
def get_exhibition_status(start_date, end_date):
    # 다양한 날짜 형식 자동 파싱: "2025년 5월 20일", "2025.05.20" 등
    # 현재 날짜와 비교하여 current/ended/upcoming/unknown 분류
```

### 2. 시간 컨텍스트 필터링 (`rag.py`)
```python
def _filter_outdated_content(self, query, items):
    # 간소화된 필터링:
    # 1. 키워드 기반: constants.PAST_INDICATORS 사용
    # 2. 연도 기반: 과거 연도(2000-2024) 언급시 점수 0.5배
    # 3. 현재 정보 우선: constants.CURRENT_INFO_KEYWORDS로 감지
```

### 3. 하이브리드 검색 (`retriever.py`)
```python
def hybrid(self, query, k_dense=30, k_bm25=50, top_k=8, rerank=True):
    # Dense search (BGE-M3) + BM25 search 결합
    # CrossEncoder로 최종 리랭킹
    # 점수 가중합: dense_score + bm25_score * 0.1
```

### 4. 멀티타겟 데이터 생성 (`data_generator.py`)
```python
def generate_responses(source_content, question):
    """하나의 소스에서 일반+어린이 응답 생성"""

    # 일반용: 전문적, 상세한 설명 (300-500자)
    general_response = ollama_api.generate(f"""
    다음 박물관 정보를 바탕으로 전문적으로 답변해주세요:
    질문: {question}
    정보: {source_content}
    """)

    # 어린이용: 친근하고 재미있는 설명 + 이모지 (100-150자)
    children_response = ollama_api.generate(f"""
    다음을 어린이가 이해하기 쉽게 설명해주세요:
    - "~이에요", "~답니다" 등 친근한 말투
    - 적절한 이모지 사용 (🌟, 🎨, ⚔️ 등)
    질문: {question}
    정보: {source_content}
    """)
```

### 5. LoRA 파인튜닝 (`fine_tuning.py`)
```python
class MultiTargetTrainer:
    def setup_lora_config(self, target_type):
        """LoRA 설정 - 효율적 파인튜닝"""
        lora_config = LoraConfig(
            r=16,                    # LoRA 랭크
            lora_alpha=32,          # 스케일링 파라미터
            target_modules=[        # 어텐션 레이어만 파인튜닝
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.1,       # 과적합 방지
        )
        # 결과: 전체 모델의 1.18%만 학습 (매우 효율적)
```

### 6. 교통 정보 정확도 개선 (`rag.py`)
```python
def _detect_transport_intent(self, query):
    """질문에서 교통수단 의도 파악"""
    # constants.TRANSPORT_KEYWORDS 사용
    for transport_type, keywords in TRANSPORT_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            return transport_type  # subway, bus, car, taxi

    # 지하철 질문 → 지하철 정보만 제공
    # 관련 문서 점수 5배 가중
```

### 7. 박물관 도메인 특화 (`constants.py`)
```python
QUERY_EXPANSION_KEYWORDS = {
    # 박물관 전문 용어 확장:
    "전시": ["전시회", "특별전", "기획전", "상설전시"],
    "국보": ["보물", "중요문화재", "문화유산"],
    "지하철": ["전철", "메트로", "지하철역", "전철역", "역", "호선"]
}
```

## 🆕 AI 도슨트 고도화 기능

### 실시간 날짜 인식
- 시스템 프롬프트에 현재 날짜 (`2025년 9월 25일`) 주입
- "현재", "이번달", "지금" 등의 질문에 시간 맥락적 답변

### 지능형 정보 우선순위
- **현재 정보 우선**: 입장료, 운영시간 등 현재 유효한 정보만 제공
- **과거 정보 억제**: 종료된 전시의 입장료나 프로그램 정보 제외
- **컨텍스트 인식**: "현재 진행 중인 특별전이 없습니다" 등 명시적 안내

### 전문 도슨트 역할
- **전시 안내**: 현재/예정/지난 전시 구분, 연령대별 맞춤 추천
- **관람 안내**: 현재 유효한 운영시간, 입장료, 교통편 정보
- **교육적 해설**: 유물의 역사적 배경, 문화적 의미, 시대별 특징 비교
- **편의 서비스**: 접근성, 편의시설, 주차, 프로그램 안내

### 박물관 특화 UI
- **빠른 질문**: 6개 카테고리별 원클릭 질문 버튼
- **시간 우선순위**: 현재/모든/과거 정보 필터 옵션
- **상세 평가**: 정확성, 충분성, 명확성, 근거성 4차원 평가
- **출처 추적**: 검색된 문서별 신뢰도 및 URL 표시

## 📊 모델 및 성능

### 🎯 **데이터 성과**
- **크롤링 데이터**: 2194개 (유물, 전시, 교통 정보 등)
- **학습 데이터 생성**: 2000개 → 539개 고품질 (중복 제거: 75.7%)
- **최종 훈련 데이터**: 485개 훈련 + 54개 검증
- **응답 품질**: 일반(457.3자), 어린이(334.6자) 평균

### 🚀 **Tesla V100 성능**
- **데이터 생성 속도**: 배치당 2-5초 (20개씩)
- **총 생성 시간**: 2000개 기준 약 2-3시간
- **LoRA 훈련 시간**: 485개 데이터로 10-15분/모델
- **GPU 사용률**: 74% 활용 (최적화됨)

### 🤖 **모델 구성**
- **기본 모델**: Qwen2.5-1.5B-Instruct
- **LoRA 파인튜닝**: 전체 파라미터의 1.18%만 훈련
- **임베딩**: BGE-M3 (한국어 최적화)
- **벡터 DB**: ChromaDB (코사인 유사도)

### 평가 메트릭
```bash
python src/eval.py --qa evaluation_qa.jsonl --k 6
# Hit@6: 검색 정확도
# MRR@6: 평균 역순위
# Faithfulness: 인용 포함률
```

## 🎯 사용 시나리오

### 관람객 질문 예시
- **전시 정보**: "현재 전시 추천해주세요", "가족과 함께 볼 전시는?"
- **관람 안내**: "오늘 운영시간은?", "입장료가 얼마인가요?"
- **유물 해설**: "국보 1호는 무엇인가요?", "고려청자 특징은?"
- **편의 서비스**: "지하철로 오는 방법은?", "어린이 체험 프로그램은?"

### AI 도슨트 답변 특징
- ✅ **시간 정확성**: 현재 유효한 정보만 제공
- ✅ **전문성**: 박물관 도메인 특화된 상세 설명
- ✅ **근거 제시**: 모든 답변에 출처 인용 포함
- ✅ **친화성**: 어린이도 이해하기 쉬운 설명

## 🔧 환경 설정

### 필수 패키지
```bash
pip install streamlit chromadb sentence-transformers rank-bm25
pip install transformers torch pydantic tqdm pdfplumber
pip install beautifulsoup4 requests
```

### 선택적 설정
```bash
# Ollama 사용시 (선택사항)
export OLLAMA_MODEL="qwen2.5:1.5b"
export OLLAMA_URL="http://localhost:11434"

# HuggingFace 모델 변경시 (기본: Qwen/Qwen2.5-1.5B-Instruct)
export HF_MODEL="Qwen/Qwen2.5-3B-Instruct"
```

## ⚙️ 시스템 제어 옵션

### 크롤링 제어
```bash
# 전시 정보 크롤링 옵션
python src/crawl/crawl.py
# - ALLOWED_HALL_IDS: 크롤링할 전시관 ID 목록
# - 상설전시 + 특별전시 자동 수집
# - 중복 제거 및 오류 처리 내장

# 해설 안내 크롤링
python src/crawl/crawl_commentary.py
# - doctype: 'web-commentary'로 자동 분류
# - 정기해설, 예약해설, 수어해설 등 수집

# 이용 안내 크롤링
python src/crawl/crawl_visitor_info.py
# - doctype: 'web-visitor'로 자동 분류
# - 관람안내, 교통정보, 편의시설 등 수집
```

### 데이터 처리 제어
```bash
# 텍스트 정제 옵션
python src/curate_docs.py [input_dir] [output_dir]
# - input_dir: 기본값 data_raw
# - output_dir: 기본값 data_curated
# - 노이즈 필터링: NOISE_LINES, NOISE_CONTAINS 사용
# - 라이선스 정보 자동 추출

# 청킹 제어
python src/clean_chunk.py [input_dir] -o [output_file]
# - input_dir: 기본값 data_curated
# - output_file: 기본값 data_curated/chunks.jsonl
# - min_chars: 최소 청크 크기 (기본 400자)
# - max_chars: 최대 청크 크기 (기본 1200자)
# - 문단 우선 → 문장 단위 분할
```

### 임베딩 & 인덱싱 제어
```bash
# ChromaDB 인덱스 생성 옵션
python src/embed_index.py \
    --chunks data_curated/chunks.jsonl \    # 청크 파일 경로
    --persist index/chroma \                 # 인덱스 저장 경로
    --collection nmK \                       # 컬렉션 이름
    --model BAAI/bge-m3 \                   # 임베딩 모델
    --chunk-batch 256 \                     # 청크 배치 크기
    --encode-batch 64 \                     # 인코딩 배치 크기
    --no-overwrite                          # 기존 컬렉션에 추가 (덮어쓰기 방지)

# 지원되는 임베딩 모델
# - BAAI/bge-m3 (기본, 다국어)
# - BAAI/bge-large-en-v1.5
# - sentence-transformers/all-MiniLM-L6-v2
```

### 검색 시스템 제어
```bash
# 하이브리드 검색 파라미터 (retriever.py)
def hybrid(self, query: str,
          k_dense=30,        # Dense 검색 결과 수
          k_bm25=50,         # BM25 검색 결과 수
          top_k=8,           # 최종 반환 결과 수
          rerank=True):      # CrossEncoder 리랭킹 사용 여부

# 시간 필터링 제어 (rag_chain.py)
def _filter_outdated_content(self, query, items):
    # current_info_keywords: 현재 정보 요청 키워드
    # past_indicators: 과거 정보 지시 키워드
    # 점수 가중치 조정 가능 (0.1 ~ 1.0)
```

### AI 도슨트 UI 제어 (Streamlit)
```bash
streamlit run app.py
# 사이드바 설정 옵션:
# - 검색 결과 개수 (k): 3~10개 슬라이더
# - 시간 우선순위: 현재/모든/과거 정보 라디오 버튼
# - 관심 분야 필터: 전시유형, 콘텐츠유형, 시대별 필터
# - 표시 옵션: 원문 보기, 문서 유형 표시
# - 시스템 제어: 세션 초기화 버튼
```

### 평가 시스템 제어
```bash
# RAG 성능 평가
python src/eval.py --qa evaluation_qa.jsonl --k 6
# - qa: 평가용 질문-답변 JSONL 파일
# - k: 검색 결과 수 (Hit@k, MRR@k 계산)
# - 메트릭: Hit@k, MRR@k, Faithfulness

# 평가 데이터 형식 (evaluation_qa.jsonl)
{"question": "현재 전시는?", "gold_urls": ["https://museum.go.kr/..."]}
```

### 로그 및 디버깅 제어
```bash
# 사용자 평가 로그 (ai_docent_evaluations.csv)
# - timestamp, question, answer, accuracy, sufficiency 등
# - 자동 저장, CSV 형식으로 누적

# 디버깅 모드
# - Streamlit: 검색된 원문 보기 체크박스
# - 신뢰도 점수 표시
# - 문서 유형별 그룹핑
# - 출처 URL 링크 제공
```

### 성능 최적화 제어
```bash
# GPU 메모리 관리
# - torch_dtype=torch.float16 (메모리 절약)
# - device_map="auto" (자동 GPU 할당)

# 배치 처리 최적화
# - chunk_batch: 인덱싱시 청크 배치 크기
# - encode_batch: 임베딩 인코딩 배치 크기
# - show_progress_bar=False (성능 최적화)

# 캐시 시스템
# - _pipe_cache: LLM 모델 캐시 (메모리 재사용)
# - ChromaDB 영구 저장 (디스크 캐시)
```

## 📈 확장 가능성

- **다국어 지원**: BGE-M3의 다국어 임베딩 활용
- **음성 인터페이스**: STT/TTS 연동 가능
- **이미지 검색**: CLIP 모델로 유물 이미지 검색
- **개인화**: 사용자 선호도 기반 추천
- **실시간 업데이트**: 웹사이트 변경사항 자동 반영

---

**🏛️ 국립중앙박물관 AI 도슨트**는 전통문화와 최신 AI 기술의 만남으로, 관람객에게 더욱 풍성하고 개인화된 박물관 경험을 제공합니다.