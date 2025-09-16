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
evaluate bert-score pandas numpy sacrebleu rouge-score tqdm

사용 예시(기본값 활용):
python week6_gpt2_stsb.py prepare
python week6_gpt2_stsb.py train
python week6_gpt2_stsb.py predict
python week6_gpt2_stsb.py evaluate --pred_csv runs/gpt2-stsb/preds_validation.csv
"""
import os
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
import torch
import evaluate  # 평가 라이브러리

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
        df = ds[split].to_pandas()[["sentence1", "sentence2"]].rename(
            columns={"sentence1": "input", "sentence2": "target"}
        )
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
    tokenizer.padding_side = "left"   # decoder-only 모델은 left padding 권장

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # HF datasets로 래핑
    ds_dict = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "validation": Dataset.from_pandas(df_val)
    })

    tokenized = ds_dict.map(
        lambda x: tokenize_function(x, tokenizer, cfg.max_length),
        batched=True,
        remove_columns=ds_dict["train"].column_names
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8  # 약간의 효율성 향상
    )

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
        evaluation_strategy="epoch",   # 최신 표기
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
# 단계 3: 추론 (배치 루프, 덮어쓰기 저장)
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
    batch_size: int = 16

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
    tokenizer.padding_side = "left"
    
    # GPU이면 FP16으로 로드하여 속도/메모리 절약
    dtype = torch.float16 if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(cfg.out_dir, torch_dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # 배치 생성
    inputs_list = df["input"].astype(str).tolist()
    prompts = [build_prompt(s) for s in inputs_list]
    gens = []
    torch.manual_seed(cfg.seed)

    for i in tqdm(range(0, len(prompts), cfg.batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+cfg.batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                temperature=cfg.temperature,
                num_return_sequences=1,  # 배치-프롬프트 매칭 단순화
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # 각 샘플에서 프롬프트를 제거하여 순수 생성만 추출
        for prompt_text, full_text in zip(batch_prompts, decoded):
            if full_text.startswith(prompt_text):
                gens.append(full_text[len(prompt_text):].strip())
            else:
                gens.append(full_text.split(TARGET_PREFIX)[-1].strip())

    # 항상 덮어쓰기 저장 (append 없음)
    out_csv = os.path.join(cfg.out_dir, f"preds_{cfg.split}.csv")
    out_df = pd.DataFrame({
        "input": df["input"],
        "reference": df["target"],
        "prediction": gens
    })
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[predict] Saved: {out_csv} (overwrite)")

# ---------------------------
# 단계 4: 평가
# ---------------------------
def evaluate_file(pred_csv: str, out_csv: Optional[str] = None):
    """BLEU-1~4, ROUGE-L, BERTScore(P/R/F1) 계산"""
    df = pd.read_csv(pred_csv)

    preds = df["prediction"].astype(str).tolist()
    refs  = df["reference"].astype(str).tolist()

    # BLEU / ROUGE / BERTScore 로더
    bleu = evaluate.load("bleu")     # sacrebleu backend
    rouge = evaluate.load("rouge")   # rouge-score backend
    bertscore = evaluate.load("bertscore")

    bleu_res = bleu.compute(predictions=preds, references=refs)
    rouge_res = rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"])
    bert_res = bertscore.compute(predictions=preds, references=refs, lang="en")

    # 샘플별 점수 산출
    from sacrebleu.metrics import BLEU as SBLEU
    sbleu = SBLEU(effective_order=True)
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    bleu1_list, bleu2_list, bleu3_list, bleu4_list = [], [], [], []
    rougeL_list = []

    for p, r in zip(preds, refs):
        def sent_bleu_ngram(n):
            sb = SBLEU(effective_order=True, max_ngram_order=n)
            return sb.sentence_score(p, [r]).score / 100.0
        bleu1_list.append(sent_bleu_ngram(1))
        bleu2_list.append(sent_bleu_ngram(2))
        bleu3_list.append(sent_bleu_ngram(3))
        bleu4_list.append(sent_bleu_ngram(4))

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
    # 코퍼스 수준 요약
    print("=== Corpus-level ===")
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
