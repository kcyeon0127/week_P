import os
from typing import Optional
import numpy as np
import pandas as pd
import evaluate 

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

if __name__ == "__main__":
    # 파라미터를 코드 내에서 직접 설정
    # predict.py 실행 후 생성된 CSV 파일 경로를 지정해야 합니다.
    evaluate_file(pred_csv="runs/gpt2-stsb/preds_validation.csv")
