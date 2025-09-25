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
  - 주요 하이퍼파라미터: `--lr`, `--batch_size`, `--max_length`, `--epochs`, `--seed`, `--devices` 등.
  - `--devices`로 사용할 GPU 개수나 ID 목록을 명시할 수 있습니다(예: `--devices 1`, `--devices 0,1`).
  - 실행 예시: `python train.py --batch_size 16 --epochs 3 --devices 1`
  - 가장 좋은 검증 F1 체크포인트는 `outputs/*.ckpt`로 저장됩니다.

## 4. 예측 결과 및 평가
- `predict_and_report.py`
  - 체크포인트를 로드해 검증/테스트 Split에서 예측을 수행합니다.
  - 결과는 CSV(`outputs/predictions.csv`)로 저장되며, 정확도와 매크로 F1을 출력합니다.
  - 실행 예시: `python predict_and_report.py --checkpoint outputs/bert-ynat-epoch.ckpt`

## 5. Streamlit 데모
- `streamlit_app.py`
  - 최신 체크포인트를 자동으로 불러와 데모 페이지를 구성합니다.
  - 페이지 상단: 예제 5개에 대해 문장, 정답, 예측, 확신도, 정답 여부 테이블 표시.
  - 페이지 하단: 입력 창과 실시간 예측 결과(테이블 + 확률 막대 그래프) 제공.
  - 실행 예시: `streamlit run streamlit_app.py`

## 실행 순서 요약
1. `python train.py ...`로 모델 학습 및 체크포인트 생성.
2. `python predict_and_report.py --checkpoint <경로>`로 예측 CSV 및 지표 산출.
3. `streamlit run streamlit_app.py`로 데모 페이지 실행.

모든 스크립트는 GPU가 있으면 자동으로 활용하며, 최초 실행 시 KLUE YNAT 데이터와 HuggingFace 모델을 다운로드합니다.
