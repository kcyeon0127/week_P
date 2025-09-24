import os
from dataclasses import dataclass
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
from utils import build_example

@dataclass
class TrainConfig:
    out_dir: str = "runs/gpt2-stsb"
    model_name: str = "gpt2"
    epochs: int = 3
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
        raise FileNotFoundError("데이터가 없습니다. 먼저 `prepare.py`로 데이터를 준비하세요.")

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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
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

if __name__ == "__main__":
    config = TrainConfig()
    train(config)
