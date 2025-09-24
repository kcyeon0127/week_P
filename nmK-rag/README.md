# 국립중앙박물관 RAG 데이터 구축 프로젝트

이 프로젝트는 국립중앙박물관 웹사이트의 데이터를 수집하고 가공하여, 검색 증강 생성(RAG) 기반의 챗봇을 위한 데이터셋을 구축하는 것을 목표로 합니다.

## 디렉토리 구조

- **/data_raw**: 크롤링을 통해 수집된 원본 웹페이지 데이터(JSON 형식)가 저장되는 곳입니다.
- **/index**: `embed_index.py`를 통해 생성된 벡터 인덱스가 저장되는 곳입니다.
- **/src**: 데이터 수집, 정제, 임베딩 등 핵심 로직을 담은 파이썬 스크립트가 위치합니다.

## 주요 스크립트 설명

### 데이터 크롤링 모듈

#### `src/crawl/crawl.py`
국립중앙박물관의 전시 관련 콘텐츠를 수집하는 메인 크롤러입니다.
- 상설전시: `ALLOWED_HALL_IDS`에 정의된 전시관 ID를 기반으로 전시실과 유물 정보 수집
- 특별전시: 현재 진행중인 특별전시 상세 정보 수집
- 지능형 탐색으로 필요한 페이지만 선별적으로 수집

#### `src/crawl/crawl_commentary.py`
전시 해설 안내 관련 페이지를 크롤링합니다.
- 정기해설, 예약해설, 수어해설 등 전시 해설 관련 콘텐츠 수집
- `doctype`이 `web-commentary`로 저장되어 일반 전시 정보와 구분

#### `src/crawl/crawl_visitor_info.py`
박물관 이용 안내 정보를 크롤링합니다.
- 관람 안내, 접근성 정보, 교통 정보 등 수집
- `doctype`이 `web-visitor`로 저장

#### `src/crawl/crawl_utils.py`
크롤링에 사용되는 공통 유틸리티 함수들을 제공합니다.
- 중복 제거, 오류 처리, 상태 관리 등의 기능

### 데이터 처리 모듈

#### `src/curate_docs.py`
크롤링된 원본 JSON 데이터의 텍스트를 정제합니다.
- 웹페이지 UI 요소 제거: "확대보기", "내려받기", "QR코드" 등
- 네비게이션 breadcrumb 및 노이즈 라인 필터링
- 구조화된 필드 라벨 정리 (전시명칭, 국적/시대, 재질 등)
- 공공누리 라이선스 정보 별도 추출

#### `src/clean_chunk.py`
정제된 문서를 RAG에 적합한 청크로 분할합니다.
- 문단 단위 분할 후 최소/최대 길이에 맞춰 동적 조정 (400-1200자)
- 긴 단락은 문장 단위로 추가 분할
- 짧은 청크는 이웃 청크와 병합하여 품질 향상
- 최종 결과를 `data_curated/chunks.jsonl`로 저장

### 검색 및 생성 모듈

#### `src/embed_index.py`
청크 데이터를 벡터 인덱스로 변환합니다.
- `BAAI/bge-m3` 다국어 임베딩 모델 사용
- ChromaDB 벡터 데이터베이스에 코사인 유사도 기반으로 저장
- 배치 처리로 메모리 효율적인 인덱싱 (기본: 256 청크/배치)

#### `src/retriever.py`
하이브리드 검색을 구현합니다.
- **밀집 검색**: BGE-M3 임베딩을 사용한 의미적 유사도 검색
- **희소 검색**: BM25 알고리즘을 사용한 키워드 매칭
- **리랭킹**: CrossEncoder를 사용해 쿼리-문서 관련도 재평가
- 검색 결과를 점수 기반으로 융합하여 최종 상위 k개 반환

#### `src/llm.py`
언어 모델 인터페이스를 제공합니다.
- **Ollama 지원**: `OLLAMA_MODEL` 환경변수로 로컬 모델 사용
- **HuggingFace 지원**: `HF_MODEL` 환경변수로 모델 지정 (기본: Qwen2.5-1.5B-Instruct)
- 컨텍스트와 함께 프롬프트 구성, 인용 형태로 답변 생성

#### `src/rag_chain.py`
전체 RAG 파이프라인을 통합 관리합니다.
- 검색과 생성을 연결하는 메인 인터페이스
- 시스템 프롬프트로 박물관 안내 챗봇 역할 정의
- 답변에서 인용 번호 추출하여 출처 정보 반환

### 평가 및 애플리케이션

#### `src/eval.py`
RAG 시스템 성능을 정량적으로 평가합니다.
- **Hit@k**: 상위 k개 검색 결과에 정답 문서가 포함된 비율
- **MRR@k**: 평균 역순위로 검색 품질 측정
- **Faithfulness**: 생성된 답변이 검색된 문서를 인용하는 비율

#### `src/schema.py`
데이터 구조를 정의합니다.
- `Doc`: 크롤링된 문서의 메타데이터와 텍스트 구조
- `Chunk`: 분할된 텍스트 청크의 구조
- ID 생성 함수 및 검증 로직

#### `app.py`
Streamlit 기반 웹 데모 애플리케이션입니다.
- RAG 파이프라인을 통한 대화형 Q&A 인터페이스
- 검색된 컨텍스트와 출처 정보 시각화
- 답변 품질에 대한 정성 평가 및 로깅 기능 (정확성, 충분성, 명확성, 근거성)
- CSV 형태로 평가 결과 저장

## 프로젝트 구조 (업데이트됨)

```
nmK-rag/
├── src/
│   ├── crawl/             # 크롤링 모듈 (새로 구조화됨)
│   │   ├── __init__.py
│   │   ├── crawl_utils.py        # 공통 크롤링 유틸리티
│   │   ├── crawl.py              # 전시 콘텐츠 크롤러
│   │   ├── crawl_commentary.py   # 해설 안내 크롤러
│   │   └── crawl_visitor_info.py # 이용 안내 크롤러
│   ├── __init__.py
│   ├── clean_chunk.py     # 문서 정제 및 청킹
│   ├── embed_index.py     # 임베딩 및 인덱싱
│   ├── eval.py            # 평가 메트릭
│   ├── llm.py             # 언어 모델 인터페이스
│   ├── parse_pdf.py       # PDF 처리
│   ├── rag_chain.py       # RAG 파이프라인
│   ├── retriever.py       # 문서 검색
│   └── schema.py          # 데이터 스키마
├── app.py                 # 웹 애플리케이션
├── data_raw/              # 크롤링된 원본 데이터 (실행 후 생성)
├── crawl_state/           # 크롤링 상태 및 로그 (실행 후 생성)
└── README.md
```

## 설정 및 실행 가이드

### 환경 요구사항
- Python 3.8+
- 메모리: 최소 8GB RAM (임베딩 모델 로딩)
- 디스크: 크롤링 데이터와 인덱스용 여유 공간

### 설치

1. **의존성 설치**:
```bash
pip install -r requirements.txt
```

2. **LLM 설정 (선택사항)**:
```bash
# Ollama 사용 시
export OLLAMA_MODEL="llama3.1:8b"
export OLLAMA_URL="http://localhost:11434"

# 또는 HuggingFace 모델 사용 시
export HF_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
```

## 전체 RAG 구축 워크플로우

### 1단계: 웹 크롤링

세 가지 전문화된 크롤러로 국립중앙박물관 데이터를 수집합니다:

```bash
# A. 전시 콘텐츠 (상설전시, 특별전시, 유물 정보)
cd src/crawl && python crawl.py

# B. 전시 해설 (정기해설, 예약해설, 수어해설)
cd src/crawl && python crawl_commentary.py

# C. 이용 안내 (관람 안내, 접근성, 교통 정보)
cd src/crawl && python crawl_visitor_info.py
```

**크롤링 특징:**
- URL과 콘텐츠 기반 중복 제거
- 중단 시 재개 가능한 상태 관리
- robots.txt 준수 및 지능적 대기
- 지수 백오프 오류 처리
- SQLite 기반 메모리 효율적 추적

**출력:** `data_raw/` 디렉토리에 JSON 파일들 생성

### 2단계: 데이터 정제

크롤링된 원본 데이터에서 웹페이지 노이즈를 제거합니다:

```bash
python src/curate_docs.py data_raw data_curated
```

- "확대보기", "내려받기" 등 UI 요소 제거
- 구조화된 필드 라벨 정리
- 공공누리 라이선스 정보 분리

**출력:** `data_curated/` 디렉토리에 정제된 JSON 파일들 생성

### 3단계: 텍스트 청킹

정제된 문서를 RAG에 적합한 청크로 분할합니다:

```bash
python src/clean_chunk.py data_curated
```

- 400-1200자 범위로 의미 단위 분할
- 문단 우선, 필요시 문장 단위 분할
- 짧은 청크는 이웃과 병합

**출력:** `data_curated/chunks.jsonl` 파일 생성

### 4단계: 벡터 인덱스 구축

청크를 임베딩하여 검색 가능한 벡터 인덱스를 생성합니다:

```bash
python src/embed_index.py
```

- BGE-M3 다국어 임베딩 모델 사용
- ChromaDB 벡터 데이터베이스 구축
- 배치 처리로 메모리 효율적 인덱싱

**출력:** `index/chroma/` 디렉토리에 벡터 인덱스 생성

### 5단계: RAG 시스템 실행

완성된 인덱스로 RAG 챗봇을 실행합니다:

```bash
streamlit run app.py
```

**기능:**
- 하이브리드 검색 (밀집 + 희소 + 리랭킹)
- 컨텍스트 기반 답변 생성 및 인용
- 검색 결과와 출처 시각화
- 답변 품질 평가 및 로깅

### 6단계: 성능 평가 (선택사항)

준비된 평가 데이터셋으로 RAG 성능을 측정합니다:

```bash
python src/eval.py --qa evaluation_qa.jsonl --k 6
```

- Hit@k, MRR@k, Faithfulness 메트릭 계산
- 검색 및 생성 품질 정량 평가

## 모니터링 및 디버깅

### 크롤링 진행 상황 확인
```bash
# 실시간 로그 확인
tail -f crawl.log
tail -f crawl_commentary.log
tail -f crawl_visitor_info.log
```

### 크롤링 통계
크롤러들은 다음과 같은 상세 통계를 제공합니다:
- 성공적으로 처리된 페이지 수
- 중복 콘텐츠/URL로 스킵된 수
- 실패한 요청 수
- 처리 속도
- 품질 필터링 결과

## 고급 설정

크롤링 동작을 유틸리티 클래스를 사용해 사용자 정의할 수 있습니다:

```python
from src.crawl.crawl_utils import CrawlConfig, URLTracker, CrawlStats

# 사용자 정의 설정
config = CrawlConfig(
    out_dir="custom_data",
    min_delay=2.0,      # 더 느린 크롤링
    max_delay=5.0,
    min_content_length=200  # 더 높은 품질 기준
)
```

## 문제 해결

### 일반적인 문제들

1. **크롤링 중단**: 같은 크롤러를 다시 실행하면 중단된 지점부터 재개됩니다
2. **속도 제한**: 크롤러가 자동으로 처리하지만, 설정에서 대기 시간을 조정할 수 있습니다
3. **메모리 문제**: 대용량 크롤링을 위해 SQLite를 사용해 메모리를 효율적으로 관리합니다
4. **콘텐츠 품질**: 필요시 `validate_content()` 함수에서 품질 기준을 조정할 수 있습니다

### 로그 및 디버깅
- 크롤링 로그: `crawl.log`, `crawl_commentary.log`, `crawl_visitor_info.log`
- 상태 파일: `crawl_state/` 디렉토리
- 데이터베이스 파일: `crawl_state/*.db`
