import os
from datasets import load_dataset

def prepare_data(out_dir: str = "data"):
    """GLUE STS-B를 다운로드하고 (train/validation/test) -> CSV로 저장"""
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("glue", "stsb")
    
    # sentence1/2와 label(유사도) 존재. sentence2를 타겟으로 사용.
    for split in ds.keys():
        df = ds[split].to_pandas()[["sentence1", "sentence2"]].rename(
            columns={"sentence1": "input", "sentence2": "target"}
        )
        df.to_csv(os.path.join(out_dir, f"stsb_{split}.csv"), index=False, encoding="utf-8")
    print(f"[prepare] Saved CSVs to: {out_dir}")

if __name__ == "__main__":
    prepare_data(out_dir="data")
