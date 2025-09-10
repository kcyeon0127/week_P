import numpy as np
import torch
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def main():
    # --- 1. KLUE-TC 데이터셋 로드 ---
    # KLUE 벤치마크의 'tc'(주제 분류) 서브셋을 사용합니다.
    # 빠른 시연을 위해 데이터셋의 작은 부분만 사용합니다. (학습 100개, 검증 50개)
    train_dataset = load_dataset("klue", "tc", split="train[:100]")
    eval_dataset = load_dataset("klue", "tc", split="validation[:50]")

    # --- 2. 토크나이저 로드 ---
    # 사용할 모델(klue/bert-base)의 이름을 지정합니다.
    model_name = "klue/bert-base"
    # 지정된 모델에 맞는 토크나이저를 불러옵니다.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- 3. 데이터 전처리 ---
    # 텍스트를 토큰화하는 함수를 정의합니다.
    def tokenize_function(examples):
        # 'title' 필드의 텍스트를 가져와 토크나이징합니다.
        # padding='max_length' : 문장 길이를 맞추기 위해 패딩을 추가합니다.
        # truncation=True : 최대 길이를 초과하는 문장을 자릅니다.
        return tokenizer(examples["title"], padding="max_length", truncation=True)

    # map 함수를 사용하여 전체 데이터셋에 토크나이징 함수를 일괄 적용합니다.
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # --- 4. 모델 로드 ---
    # 데이터셋의 레이블 개수를 확인합니다.
    num_labels = train_dataset.features["label"].num_classes
    # 사전 학습된 모델을 로드하면서, 분류할 레이블의 개수를 지정해줍니다.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # --- 5. 평가지표 정의 ---
    # 'accuracy' 평가지표를 로드합니다.
    metric = evaluate.load("accuracy")

    # 평가 시 모델의 예측 결과를 어떻게 계산할지 정의하는 함수입니다.
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # 로짓(logits)에서 가장 높은 값을 가진 인덱스를 예측값으로 사용합니다.
        predictions = np.argmax(logits, axis=-1)
        # 예측값과 실제 레이블을 비교하여 정확도를 계산합니다.
        return metric.compute(predictions=predictions, references=labels)

    # --- 6. 학습 인자(Training Arguments) 정의 ---
    training_args = TrainingArguments(
        output_dir="./W_5/results",          # 모델 체크포인트와 로그가 저장될 디렉토리
        num_train_epochs=1,              # 총 학습 에폭 수 (빠른 테스트를 위해 1로 설정)
        per_device_train_batch_size=8,   # 학습용 배치 사이즈
        per_device_eval_batch_size=8,    # 평가용 배치 사이즈
        warmup_steps=10,                 # 학습률 스케줄러를 위한 웜업 스텝 수
        weight_decay=0.01,               # 가중치 감소 강도
        logging_dir="./W_5/logs",            # 로그가 저장될 디렉토리
        logging_steps=10,                # 로그를 기록할 스텝 간격
        evaluation_strategy="epoch",     # 매 에폭이 끝날 때마다 평가를 수행
    )

    # --- 7. Trainer 초기화 ---
    # 학습을 실제로 수행할 Trainer 객체를 초기화합니다.
    trainer = Trainer(
        model=model,                         # 파인튜닝할 모델
        args=training_args,                  # 위에서 정의한 학습 인자
        train_dataset=tokenized_train_dataset, # 학습 데이터셋
        eval_dataset=tokenized_eval_dataset,   # 평가 데이터셋
        compute_metrics=compute_metrics,     # 평가지표 계산 함수
    )

    # --- 8. 학습 및 평가 ---
    print("--- 파인튜닝 시작 ---")
    # 모델 학습을 시작합니다.
    trainer.train()

    print("\n--- 파인튜닝된 모델 평가 ---")
    # 학습이 완료된 모델을 평가합니다.
    eval_results = trainer.evaluate()
    print(f"평가 정확도: {eval_results['eval_accuracy']:.4f}")

# 이 스크립트가 직접 실행될 때 main 함수를 호출합니다.
if __name__ == "__main__":
    main()
