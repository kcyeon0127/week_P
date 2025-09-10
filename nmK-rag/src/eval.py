import json, argparse
from typing import List, Dict
from statistics import mean
from src.rag_chain import RAG

def evaluate(rag: RAG, qa_path: str, k: int = 6):
    gold = [json.loads(l) for l in open(qa_path,"r",encoding="utf-8")]
    hits, mrrs, faith = [], [], []
    for ex in gold:
        q = ex["question"]
        gold_urls = set(ex.get("gold_urls", []))
        ctx = rag.retrieve(q, k=k)
        # Hit@k / Recall@k
        ctx_urls = [c["url"] for c in ctx if c["url"]]
        hit = 1.0 if any(u in gold_urls for u in ctx_urls) else 0.0
        hits.append(hit)
        # MRR
        rr = 0.0
        for i, u in enumerate(ctx_urls):
            if u in gold_urls:
                rr = 1.0 / (i+1)
                break
        mrrs.append(rr)
        # Faithfulness: 생성 결과에 인용이 1개 이상 포함됐는지
        ans = rag.generate(q, ctx)
        faith.append(1.0 if ans.citations else 0.0)
    print(f"Hit@{k}: {mean(hits):.3f} | MRR@{k}: {mean(mrrs):.3f} | Faithfulness(rate of citation): {mean(faith):.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="jsonl with fields: question, gold_urls[]")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    rag = RAG()
    evaluate(rag, args.qa, k=args.k)
