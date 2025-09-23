import json, argparse, re, sys
from pathlib import Path
from typing import List, Tuple

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.schema import Doc, Chunk, make_id
from src.curate_docs import clean_doc
from tqdm import tqdm

DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "data_curated"
DEFAULT_OUTPUT_FILE = DEFAULT_INPUT_DIR / "chunks.jsonl"

def split_into_chunks(text: str, min_chars=400, max_chars=1200) -> List[Tuple[str, int, int]]:
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
                sents = re.split(r"(?<=[.!?…])\s+", p)
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
    chunks_with_range: List[Tuple[str, int, int]] = []
    search_pos = 0
    for chunk_text in merged:
        idx = text.find(chunk_text, search_pos)
        if idx == -1:
            idx = text.find(chunk_text)
            if idx == -1:
                idx = search_pos
        start = idx
        end = idx + len(chunk_text)
        chunks_with_range.append((chunk_text, start, end))
        search_pos = end
    return chunks_with_range

def main(in_dir: str, out_jsonl: str):
    files = sorted(Path(in_dir).rglob("*.json"))
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w", encoding="utf-8") as w:
        for fp in tqdm(files):
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
            doc = Doc(**data)
            doc = clean_doc(doc)
            if not doc.text.strip():
                continue
            chunks = split_into_chunks(doc.text)
            for i, (txt, start, end) in enumerate(chunks):
                ch = Chunk(
                    chunk_id = make_id(f"{doc.doc_id}-{i}"),
                    doc_id = doc.doc_id,
                    title = doc.title,
                    url = doc.url,
                    section = doc.section,
                    doctype = doc.doctype,
                    lang = doc.lang,
                    chunk_index = i,
                    char_range = [start, end],
                    text = txt,
                    meta = {"source": str(fp)}
                )
                w.write(ch.model_dump_json(ensure_ascii=False) + "\n")
                total += 1
    print(f"[done] wrote {total} chunks to {out_jsonl}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir", nargs="?", default=str(DEFAULT_INPUT_DIR), help="data_curated or data_raw/web etc.")
    ap.add_argument("-o", "--out", default=str(DEFAULT_OUTPUT_FILE))
    args = ap.parse_args()
    main(args.in_dir, args.out)
