#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   1. 데이터 준비:
   1     python run.py prepare

   2. 모델 학습:
   1     python run.py train

   3. 텍스트 생성 (추론):
   1     python run.py predict

   4. 결과 평가:

   1     python run.py evaluate --pred_csv runs/gpt2-stsb/preds_validation.csv
"""
import argparse
from prepare import prepare_data
from train import train, TrainConfig
from predict import predict, GenConfig
from evaluate import evaluate_file

def build_parser():
    p = argparse.ArgumentParser(description="Week6 GPT-2 Text Generation (GLUE STS-B)")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="데이터 다운로드 및 CSV 저장")
    sp.add_argument("--out_dir", type=str, default="data")

    st = sub.add_parser("train", help="GPT-2 학습")
    st.add_argument("--out_dir", type=str, default="runs/gpt2-stsb")
    st.add_argument("--model_name", type=str, default="gpt2")
    st.add_argument("--epochs", type=int, default=1)
    st.add_argument("--batch_size", type=int, default=8)
    st.add_argument("--lr", type=float, default=5e-5)
    st.add_argument("--weight_decay", type=float, default=0.01)
    st.add_argument("--warmup_ratio", type=float, default=0.03)
    st.add_argument("--max_length", type=int, default=256)
    st.add_argument("--seed", type=int, default=42)
    st.add_argument("--grad_accum", type=int, default=1)
    st.add_argument("--no_fp16", action="store_true", help="FP16 비활성화")

    spd = sub.add_parser("predict", help="생성 추론 (배치, 항상 덮어쓰기 저장)")
    spd.add_argument("--out_dir", type=str, default="runs/gpt2-stsb")
    spd.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    spd.add_argument("--max_new_tokens", type=int, default=64)
    spd.add_argument("--top_k", type=int, default=50)
    spd.add_argument("--top_p", type=float, default=0.95)
    spd.add_argument("--temperature", type=float, default=0.8)
    spd.add_argument("--num_return_sequences", type=int, default=1)
    spd.add_argument("--seed", type=int, default=42)
    spd.add_argument("--batch_size", type=int, default=16)

    se = sub.add_parser("evaluate", help="CSV 기반 평가 (BLEU/ROUGE/BERTScore)")
    se.add_argument("--pred_csv", type=str, required=True, help="predict 단계에서 저장된 CSV 경로")
    se.add_argument("--out_csv", type=str, default=None)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "prepare":
        prepare_data(args.out_dir)

    elif args.cmd == "train":
        cfg = TrainConfig(
            out_dir=args.out_dir,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_length=args.max_length,
            seed=args.seed,
            gradient_accumulation_steps=args.grad_accum,
            fp16=not args.no_fp16
        )
        train(cfg)

    elif args.cmd == "predict":
        cfg = GenConfig(
            out_dir=args.out_dir,
            split=args.split,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            num_return_sequences=args.num_return_sequences,
            seed=args.seed,
            batch_size=args.batch_size
        )
        predict(cfg)

    elif args.cmd == "evaluate":
        evaluate_file(args.pred_csv, args.out_csv)

if __name__ == "__main__":
    main()
