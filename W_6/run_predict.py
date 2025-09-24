import os
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils import build_prompt, TARGET_PREFIX
from utils_checkpoint import find_latest_checkpoint

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
        raise FileNotFoundError(f"{csv_path} 가 없습니다. 먼저 `prepare.py`를 실행하세요.")

    df = pd.read_csv(csv_path)

    # 모델/토크나이저 - 최신 체크포인트 자동 찾기
    model_path = find_latest_checkpoint(cfg.out_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # GPU이면 FP16으로 로드하여 속도/메모리 절약
    dtype = torch.float16 if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
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
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
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

if __name__ == "__main__":
    config = GenConfig()
    predict(config)
