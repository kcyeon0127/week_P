# Week 7: BERT 기반 KLUE YNAT 분류 프로젝트

이 디렉터리는 KLUE YNAT 뉴스 토픽 분류(N→1)의 전체 파이프라인을 담고 있습니다. HuggingFace 사전학습 BERT 가중치를 직접 복사한 백본을 사용하고, PyTorch Lightning으로 학습/평가하며 Streamlit으로 데모를 제공합니다.

## 1. 모델 구현
- `models.py`
  - `CustomBertBackbone`: HuggingFace `bert-base-multilingual-cased`에서 가중치를 복사해 동일 구조의 BERT 인코더를 생성합니다.
  - `LightningBertClassifier`: BERT 백본 위에 분류 헤드를 얹은 Lightning 모듈로, 매크로 Accuracy/F1을 로깅합니다.

## 2. 태스크 및 데이터
- 태스크: KLUE YNAT 뉴스 토픽 분류 (N개의 토큰 입력 → 1개의 라벨 출력, N→1).
- 데이터 로더는 `datamodule.py`의 `YNATDataModule`이 담당하며, KLUE YNAT를 로드해 학습/검증/테스트 분할과 토크나이징을 수행합니다.

## 3. 학습 파이프라인
- `train.py`
  - PyTorch Lightning 학습 엔트리 포인트입니다.
  - 학습률, 배치 크기, 에폭, GPU 장치 등은 파일 상단의 상수로 관리하며 필요하면 값을 수정한 뒤 `python train.py`만 실행하면 됩니다.
  - 기본값은 단일 GPU 사용(`devices = 1`)이며, 여러 GPU를 쓰려면 리스트로 지정하면 됩니다.
  - 가장 좋은 검증 F1 체크포인트는 `outputs/*.ckpt`로 저장됩니다.

## 4. 예측 결과 및 평가
- `predict_and_report.py`
  - `outputs/` 디렉터리에서 최신 체크포인트를 자동으로 찾아 예측을 수행하고 `outputs/predictions.csv`에 저장합니다.
  - 정확도와 매크로 F1을 함께 출력합니다. 실행은 `python predict_and_report.py` 한 줄이면 충분합니다.
- `analyze_predictions.py`
  - `outputs/predictions.csv`를 기본 경로로 읽어 정확도/매크로 F1, 라벨별 리포트, 혼동행렬 이미지를 저장합니다.
  - 추가로 낮은 확신도/오분류 샘플을 CSV로 내보냅니다. 실행 명령: `python analyze_predictions.py`

## 5. Streamlit 데모
- `streamlit_app.py`
  - 최신 체크포인트를 자동으로 불러와 데모 페이지를 구성합니다.
  - 페이지 상단: 예제 5개에 대해 문장, 정답, 예측, 확신도, 정답 여부 테이블 표시.
  - 페이지 하단: 입력 창과 실시간 예측 결과(테이블 + 확률 막대 그래프) 제공.
  - 실행 예시: `streamlit run streamlit_app.py`

## 실행 순서 요약
1. `python train.py ...`로 모델 학습 및 체크포인트 생성.
2. `python predict_and_report.py --checkpoint <경로>`로 예측 CSV 및 지표 산출.
3. `python analyze_predictions.py`로 성능 요약, 혼동행렬, 저확신/오분류 샘플을 저장.
4. `streamlit run streamlit_app.py`로 데모 페이지 실행.

> 참고: 재현 가능한 분석을 위해 `analyze_predictions.py`처럼 `.py` 스크립트로 통계를 남기는 방식을 추천합니다. 필요 시 Jupyter 노트북을 추가로 사용해 시각화나 탐색형 분석을 진행할 수 있습니다.

모든 스크립트는 GPU가 있으면 자동으로 활용하며, 최초 실행 시 KLUE YNAT 데이터와 HuggingFace 모델을 다운로드합니다.
