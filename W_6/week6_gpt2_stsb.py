#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Week 6: GPT-2 기반 Text Generation 전체 코드 (GLUE STS-B 사용)
- 학습: sentence1 -> sentence2 생성 (Causal LM 포맷)
- 디코딩: Top-k / Top-p / Temperature 설정 가능
- 평가: BLEU-1~4, ROUGE-L, BERTScore
- 결과: [입력, 정답, 생성, BLEU1~4, ROUGE_L, BERTScore_P/R/F1] CSV 저장

필수 패키지(예시):
pip install -U "transformers>=4.41.0" "datasets>=2.19.0" "accelerate>=0.33.0" \
evaluate bert-score pandas numpy sacrebleu rouge-score

사용 예시:
python week6_gpt2_stsb.py prepare --out_dir data
python week6_gpt2_stsb.py train --out_dir runs/gpt2-stsb --model_name gpt2 --epochs 2 --batch_size 8
python week6_gpt2_stsb.py predict --out_dir runs/gpt2-stsb --split validation --max_new_tokens 64 --top_k 50 --top_p 0.95 --temperature 0.8
python week6_gpt2_stsb.py evaluate --pred_csv runs/gpt2-stsb/preds_validation.csv --out_csv runs/gpt2-stsb/eval_validation.csv
"""
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
import torch

# 평가 라이브러리
import evaluate

# ---------------------------
# 공통 유틸
# ---------------------------
PROMPT_PREFIX = "Source: "
TARGET_PREFIX = "\nTarget: "
EOS = ""  # GPT-2 eos token

def build_example(s1: str, s2: str) -> str:
    """sentence1을 조건, sentence2를 타겟으로 하는 단일 텍스트 시퀀스 구성"""
    return f"{PROMPT_PREFIX}{s1}{TARGET_PREFIX}{s2}{EOS}"

def build_prompt(s1: str) -> str:
    """추론 시 조건 프롬프트"""
    return f"{PROMPT_PREFIX}{s1}{TARGET_PREFIX}"

# ---------------------------
# 단계 1: 데이터 준비
# ---------------------------
def prepare_data(out_dir: str = "data"):
    """GLUE STS-B를 다운로드하고 (train/validation/test) -> CSV로 저장"""
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("glue", "stsb")
    # sentence1/2와 label(유사도) 존재. 여기서는 sentence2를 타겟으로 사용.
    for split in ds.keys():
        df = ds[split].to_pandas()[["sentence1", "sentence2"]].rename(columns={"sentence1": "input", "sentence2": "target"})
        df.to_csv(os.path.join(out_dir, f"stsb_{split}.csv"), index=False, encoding="utf-8")
    print(f"[prepare] Saved CSVs to: {out_dir}")

# ---------------------------
# 단계 2: 학습 파이프라인
# ---------------------------
@dataclass
class TrainConfig:
    out_dir: str = "runs/gpt2-stsb"
    model_name: str = "gpt2"
    epochs: int = 1
    batch_size: int = 8
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_length: int = 256
    seed: int = 42
    gradient_accumulation_steps: int = 1
    fp16: bool = True

def tokenize_function(examples, tokenizer, max_length: int):
    texts = [build_example(s1, s2) for s1, s2 in zip(examples["input"], examples["target"])]
    out = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    # causal LM: labels = input_ids 그대로
    out["labels"] = out["input_ids"].copy()
    return out

def train(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 데이터 로드
    train_csv = os.path.join("data", "stsb_train.csv")
    val_csv   = os.path.join("data", "stsb_validation.csv")
    if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
        raise FileNotFoundError("데이터가 없습니다. 먼저 `prepare` 서브커맨드로 데이터를 준비하세요.")

    df_train = pd.read_csv(train_csv)
    df_val   = pd.read_csv(val_csv)

    # 토크나이저/모델
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    # GPT-2는 pad_token이 없음 -> eos로 채우기
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # HF datasets로 래핑
    ds_dict = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "validation": Dataset.from_pandas(df_val)
    })

    tokenized = ds_dict.map(lambda x: tokenize_function(x, tokenizer, cfg.max_length), batched=True, remove_columns=ds_dict["train"].column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 학습 설정
    training_args = TrainingArguments(
        output_dir=cfg.out_dir,
        overwrite_output_dir=True,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        seed=cfg.seed,
        fp16=cfg.fp16,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)
    print(f"[train] Finished. Model saved to: {cfg.out_dir}")

# ---------------------------
# 단계 3: 추론
# ---------------------------
@dataclass
class GenConfig:
    out_dir: str = "runs/gpt2-stsb"
    split: str = "validation"  # train/validation/test
    max_new_tokens: int = 64
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 0.8
    num_return_sequences: int = 1
    seed: int = 42

def predict(cfg: GenConfig):
    # 데이터 로드
    csv_path = os.path.join("data", f"stsb_{cfg.split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} 가 없습니다. 먼저 `prepare`를 실행하세요.")

    df = pd.read_csv(csv_path)

    # 모델/토크나이저
    tokenizer = AutoTokenizer.from_pretrained(cfg.out_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.out_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    gens = []
    torch.manual_seed(cfg.seed)
    for idx, row in df.iterrows():
        prompt = build_prompt(row["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                temperature=cfg.temperature,
                num_return_sequences=cfg.num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # prompt를 제거한 생성 부분만 추출
        if decoded.startswith(prompt):
            generated = decoded[len(prompt):].strip()
        else:
            # 안전장치: 혹시 프리픽스와 정확히 매치 안 될 때
            generated = decoded.split(TARGET_PREFIX)[-1].strip()
        gens.append(generated)

    out_csv = os.path.join(cfg.out_dir, f"preds_{cfg.split}.csv")
    out_df = pd.DataFrame({
        "input": df["input"],
        "reference": df["target"],
        "prediction": gens
    })
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[predict] Saved: {out_csv}")

# ---------------------------
# 단계 4: 평가
# ---------------------------
def evaluate_file(pred_csv: str, out_csv: Optional[str] = None):
    """BLEU-1~4, ROUGE-L, BERTScore(P/R/F1) 계산"""
    df = pd.read_csv(pred_csv)

    preds = df["prediction"].astype(str).tolist()
    refs  = df["reference"].astype(str).tolist()

    # BLEU (1~4)
    bleu = evaluate.load("bleu")     # sacrebleu backend
    rouge = evaluate.load("rouge")   # rouge-score backend
    # BERTScore
    bertscore = evaluate.load("bertscore")

    # BLEU-1~4 계산을 위해 각 gram별 weights 사용
    def bleu_score(preds, refs, weights):
        # evaluate의 bleu는 누적 BLEU를 제공 -> weights로 n-gram 지정
        res = bleu.compute(predictions=preds, references=refs, max_order=len(weights), use_effective_order=True)
        # res['precisions'][0]은 1-gram precision 등. 누적 BLEU가 아니라면 직접 계산이 복잡해짐.
        # 간단히 sacrebleu BLEU(최대 4-gram) 한 번만 사용하고, 1~4gram precision을 별도로 표시.
        return res

    bleu_res = bleu.compute(predictions=preds, references=refs)
    rouge_res = rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"])
    bert_res = bertscore.compute(predictions=preds, references=refs, lang="en")

    # 결과 프레임 구성
    # evaluate의 BLEU는 corpus BLEU를 반환 -> 샘플별 점수를 원하면 별도 루프 필요.
    # 여기서는 샘플별 metric을 위해 간단한 방식으로 개별 계산(속도는 느리지만 Week6 과제 규모에 충분).
    from sacrebleu.metrics import BLEU as SBLEU
    sbleu = SBLEU(effective_order=True)
    bleu1_list, bleu2_list, bleu3_list, bleu4_list = [], [], [], []
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for p, r in zip(preds, refs):
        # sacrebleu는 corpus 기반이라 문장별은 리스트 크기 1로 처리
        bleu_scores = sbleu.sentence_score(p, [r])
        # sacrebleu는 기본적으로 1~4-gram을 함께 사용 ->
        # 간단히 동일 점수를 BLEU-1~4로 중복 표기하지 않고, 아래에서는 모두 동일 값 대신
        # precision(n)을 근사로 사용하기보다는 통일된 문장 BLEU를 BLEU-4로 두고
        # BLEU-1~3은 n-gram 수를 강제로 제한한 BLEU를 추가 계산
        def sent_bleu_ngram(n):
            sb = SBLEU(effective_order=True, max_ngram_order=n)
            return sb.sentence_score(p, [r]).score / 100.0

        bleu1_list.append(sent_bleu_ngram(1))
        bleu2_list.append(sent_bleu_ngram(2))
        bleu3_list.append(sent_bleu_ngram(3))
        bleu4_list.append(sent_bleu_ngram(4))

    rougeL_list = []
    for p, r in zip(preds, refs):
        score = rs.score(r, p)  # (target, prediction)
        rougeL_list.append(score["rougeL"].fmeasure)

    # BERTScore는 문장별 점수 제공
    bert_P = [v for v in bert_res["precision"]]
    bert_R = [v for v in bert_res["recall"]]
    bert_F1 = [v for v in bert_res["f1"]]

    out_df = pd.DataFrame({
        "input": df["input"],
        "reference": refs,
        "prediction": preds,
        "BLEU-1": bleu1_list,
        "BLEU-2": bleu2_list,
        "BLEU-3": bleu3_list,
        "BLEU-4": bleu4_list,
        "ROUGE-L": rougeL_list,
        "BERTScore_P": bert_P,
        "BERTScore_R": bert_R,
        "BERTScore_F1": bert_F1,
    })

    if out_csv is None:
        out_csv = os.path.join(os.path.dirname(pred_csv), f"eval_{os.path.basename(pred_csv).replace('preds_', '')}")
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[evaluate] Saved: {out_csv}")
    # 코퍼스 수준 요약도 함께 출력
    print("=== Corpus-level (reference) ===")
    print(f"BLEU (sacrebleu): {bleu_res['bleu']:.4f}")
    print(f"ROUGE-L (avg):   {np.mean(rougeL_list):.4f}")
    print(f"BERTScore F1:    {np.mean(bert_F1):.4f}")

# ---------------------------
# CLI
# ---------------------------
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

    spd = sub.add_parser("predict", help="생성 추론")
    spd.add_argument("--out_dir", type=str, default="runs/gpt2-stsb")
    spd.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    spd.add_argument("--max_new_tokens", type=int, default=64)
    spd.add_argument("--top_k", type=int, default=50)
    spd.add_argument("--top_p", type=float, default=0.95)
    spd.add_argument("--temperature", type=float, default=0.8)
    spd.add_argument("--num_return_sequences", type=int, default=1)
    spd.add_argument("--seed", type=int, default=42)

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
            seed=args.seed
        )
        predict(cfg)

    elif args.cmd == "evaluate":
        evaluate_file(args.pred_csv, args.out_csv)

if __name__ == "__main__":
    main()
