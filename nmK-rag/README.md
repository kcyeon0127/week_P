# 국립중앙박물관 RAG 데이터 구축 프로젝트

이 프로젝트는 국립중앙박물관 웹사이트의 데이터를 수집하고 가공하여, 검색 증강 생성(RAG) 기반의 챗봇을 위한 데이터셋을 구축하는 것을 목표로 합니다.

## 디렉토리 구조

- **/data_raw**: 크롤링을 통해 수집된 원본 웹페이지 데이터(JSON 형식)가 저장되는 곳입니다.
- **/index**: `embed_index.py`를 통해 생성된 벡터 인덱스가 저장되는 곳입니다.
- **/src**: 데이터 수집, 정제, 임베딩 등 핵심 로직을 담은 파이썬 스크립트가 위치합니다.

## 스크립트 설명

### `src/crawl.py`

국립중앙박물관 웹사이트의 '전시' 관련 콘텐츠를 지능적으로 크롤링하는 스크립트입니다.

#### 주요 기능

- **지능형 탐색**: 단순 링크를 따라가는 것이 아니라, 사전에 정의된 규칙에 따라 필요한 페이지만 선별적으로 탐색하고 수집합니다.
  - **상설 전시**: `ALLOWED_HALL_IDS`에 정의된 전시관 ID(`showHallId`)를 기반으로 모든 하위 전시실과 그 안의 전시품(`relicId`) 상세 페이지를 수집합니다.
  - **특별 전시**: `exhiSpThemId`가 포함된 링크를 따라가 현재 진행중인 특별 전시의 상세 정보를 수집합니다.
- **Polite Crawling**: 서버에 부담을 주지 않기 위해 각 요청 사이에 1초의 지연 시간을 둡니다.

#### 설정 방법

스크립트 상단의 `[설정 영역]`에서 아래 변수들을 수정하여 크롤링 대상을 제어할 수 있습니다.

- `SEED_URLS`: 크롤링을 시작할 최상위 페이지 목록입니다. (예: 상설전시 층별안내, 현재 전시 등)
- `ALLOWED_HALL_IDS`: 수집을 허용할 상설 전시관의 고유 ID 목록입니다.

### `src/crawl_commentary.py`

'전시 해설 안내' 관련 페이지들을 크롤링하기 위해 추가된 스크립트입니다.

- **주요 기능**:
  - '정기해설', '예약해설', '수어해설' 등 '전시 해설'과 관련된 여러 페이지를 시작점으로 하여 관련 콘텐츠를 수집합니다.
  - URL 경로에 `/M0102`가 포함된 링크를 따라가며 '전시 해설'과 관련된 페이지만을 선별적으로 수집합니다.
- **데이터 구분**:
  - 이 스크립트로 수집된 데이터는 `doctype`이 `web-commentary`로 저장되어, `crawl.py`가 수집하는 전시 정보(`web`)와 구분됩니다.

### `src/clean_chunk.py`

`data_raw`에 저장된 원본 JSON 파일들을 읽어와 RAG 모델이 사용하기 좋은 형태로 가공하는 스크립트입니다.

- **주요 기능**:
  - `split_into_chunks` 함수를 통해 문서를 의미있는 단위의 청크(Chunk)로 분할합니다.
  - 문단 단위로 먼저 분할하고, 청크의 최소/최대 길이에 맞춰 동적으로 단락을 합치거나 문장 단위로 나누는 정교한 로직을 사용합니다.
  - 최종 결과물은 `data_curated/chunks.jsonl` 파일로 저장됩니다.

### `src/embed_index.py`

`clean_chunk.py`를 통해 생성된 `chunks.jsonl` 파일을 읽어, 각 텍스트 청크를 벡터로 변환하고 이를 벡터 데이터베이스에 저장하여 인덱스를 생성합니다.

- **주요 기능**:
  - `sentence-transformers` 라이브러리(`BAAI/bge-m3` 모델)를 사용하여 텍스트 임베딩을 생성합니다.
  - `ChromaDB`를 벡터 데이터베이스로 사용하며, `index/chroma` 디렉토리에 인덱스를 저장합니다.
  - 스크립트 실행 시 기존 컬렉션이 있다면 삭제하고 새로 생성하여 항상 최신 상태의 인덱스를 유지합니다.

### `app.py`

`Streamlit`으로 제작된 RAG 챗봇 데모 웹 애플리케이션입니다. 사용자가 질문을 입력하고, RAG 파이프라인을 통해 생성된 답변과 근거를 확인할 수 있습니다.

- **주요 기능**:
  - `src.rag_chain`의 `RAG` 클래스를 사용하여 검색-생성 파이프라인을 실행합니다.
  - 답변의 근거가 된 컨텍스트와 원문 출처(URL)를 함께 표시합니다.
  - 답변에 대한 정확성, 충분성 등을 평가하고 코멘트를 저장하는 기능을 포함합니다.

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

### 전제 조건
- Python 3.8+
- 필수 패키지: `requests`, `beautifulsoup4`, `tenacity`, `tqdm`, `pydantic`, `sqlite3`

### 설치

1. **의존성 설치**:
```bash
pip install requests beautifulsoup4 tenacity tqdm pydantic streamlit
```

## 전체 데이터 구축 워크플로우

### 1단계: 웹 크롤링 (개선된 버전)

이제 세 가지 전문화된 크롤러를 제공합니다:

#### A. 전시 콘텐츠 크롤러
상설전시, 특별전시, 유물 정보를 수집합니다.

```bash
cd src/crawl
python crawl.py
```

#### B. 전시 해설 크롤러
전시 해설, 가이드 투어, 접근성 정보를 수집합니다.

```bash
cd src/crawl
python crawl_commentary.py
```

#### C. 이용 안내 크롤러
관람 안내, 접근성 정보, 교통 정보를 수집합니다.

```bash
cd src/crawl
python crawl_visitor_info.py
```

#### 🔥 개선된 크롤링 기능들:
- **스마트 중복 제거**: URL과 콘텐츠 기반 중복 검사
- **재개 가능한 크롤링**: 중단되어도 이어서 진행 가능
- **속도 제한**: robots.txt 준수 및 지능적 대기
- **견고한 오류 처리**: 지수 백오프 자동 재시도
- **품질 필터링**: 저품질 콘텐츠 자동 제외
- **포괄적 로깅**: 모니터링 및 디버깅용 상세 로그
- **메모리 효율성**: 대규모 크롤링을 위한 SQLite 기반 추적

#### 크롤링 출력:
- 원본 데이터: `data_raw/` 디렉토리
- 크롤링 상태 및 로그: `crawl_state/` 디렉토리
- 각 문서는 메타데이터와 함께 JSON으로 저장

### 2단계: 데이터 정제 및 분할
```bash
python src/clean_chunk.py data_raw
```
- `data_raw` 폴더의 모든 JSON을 처리하여 `data_curated/chunks.jsonl` 파일 생성

### 3단계: 임베딩 및 인덱스 생성
```bash
python src/embed_index.py
```
- `chunks.jsonl` 파일을 읽어 `index/chroma` 디렉토리에 벡터 인덱스 생성

### 4단계: 애플리케이션 실행
```bash
streamlit run app.py
```
- 웹 인터페이스에서 RAG 기반 질의응답 시스템 사용

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
