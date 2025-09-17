#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
전체 파이프라인 실행 - 기존 파일들의 main 함수 호출
"""

def main():
    print("=== 전체 파이프라인 시작 ===")

    # 1. 데이터 준비
    print("\n1. 데이터 준비 중...")
    import run_prepare
    run_prepare.prepare_data(out_dir="data")

    # 2. 모델 훈련
    print("\n2. 모델 훈련 중...")
    import run_train
    config = run_train.TrainConfig()
    run_train.train(config)

    # 3. 예측 생성
    print("\n3. 예측 생성 중...")
    import run_predict
    gen_config = run_predict.GenConfig()
    run_predict.predict(gen_config)

    # 4. 평가
    print("\n4. 평가 중...")
    import run_evaluate
    run_evaluate.evaluate_file(pred_csv="runs/gpt2-stsb/preds_validation.csv")

    print("\n=== 모든 단계 완료 ===")

if __name__ == "__main__":
    main()