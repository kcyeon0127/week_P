import os, glob, json, argparse, re
from typing import List
from src.schema import Doc, Chunk, make_id
from tqdm import tqdm

def split_into_chunks(text: str, min_chars=400, max_chars=1200) -> List[str]:
    # 문단 단위로 먼저 자르고 합치기
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        candidate = (cur + "\n\n" + p).strip() if cur else p
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            if len(p) <= max_chars:
                cur = p
            else:
                # 너무 긴 단락은 문장 단위로 추가 분할
                sents = re.split(r"(?<=[.!?]|[.!?]\”) +", p)
                buf = ""
                for s in sents:
                    if len(buf) + len(s) + 1 <= max_chars:
                        buf = (buf + " " + s).strip()
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = s
                if buf: cur = buf
                else: cur = ""
    if cur: chunks.append(cur)
    # 짧은 조각은 이웃과 병합
    merged=[]
    for c in chunks:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = (merged[-1] + "\n\n" + c).strip()
        else:
            merged.append(c)
    return merged

def main(in_dir: str, out_jsonl: str):
    files = glob.glob(os.path.join(in_dir, "*.json"))
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    w = open(out_jsonl, "w", encoding="utf-8")
    total = 0
    for fp in tqdm(files):
        data = json.load(open(fp, "r", encoding="utf-8"))
        doc = Doc(**data)
        chunks = split_into_chunks(doc.text)
        for i, txt in enumerate(chunks):
            ch = Chunk(
                chunk_id = make_id(f"{doc.doc_id}-{i}"),
                doc_id = doc.doc_id,
                title = doc.title,
                url = doc.url,
                section = doc.section,
                doctype = doc.doctype,
                lang = doc.lang,
                chunk_index = i,
                char_range = [0, len(txt)],
                text = txt,
                meta = {"source": fp}
            )
            w.write(ch.model_dump_json(ensure_ascii=False) + "\n")
            total += 1
    w.close()
    print(f"[done] wrote {total} chunks to {out_jsonl}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir", help="data_curated or data_raw/web etc.")
    ap.add_argument("-o", "--out", default="data_curated/chunks.jsonl")
    args = ap.parse_args()
    main(args.in_dir, args.out)
