"""Build a Chroma vector index from curated chunks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEF_EMB = "BAAI/bge-m3"  # 한국어/다국어 임베딩
PASSAGE_PROMPT = "passage"


def _batched(items: Iterable[dict], size: int) -> Iterator[List[dict]]:
    buf: List[dict] = []
    for item in items:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def main(
    chunks_jsonl: str,
    persist_dir: str,
    collection: str,
    emb_model: str = DEF_EMB,
    chunk_batch: int = 256,
    encode_batch: int = 64,
    overwrite: bool = True,
) -> None:
    chunks_path = Path(chunks_jsonl)
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_jsonl}")

    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(allow_reset=overwrite),
    )

    if overwrite:
        try:
            client.delete_collection(collection)
        except Exception:
            pass

    col = client.create_collection(
        collection,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": emb_model,
            "embedding_prompt": PASSAGE_PROMPT,
        },
    )

    model = SentenceTransformer(emb_model, trust_remote_code=True)

    with chunks_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    total = len(records)
    for batch in tqdm(list(_batched(records, chunk_batch)), desc="index", unit="batch"):
        ids = [item["chunk_id"] for item in batch]
        texts = [item["text"] for item in batch]
        metas = [
            {
                "doc_id": item["doc_id"],
                "title": item["title"],
                "url": item.get("url") or "",
                "doctype": item.get("doctype") or "web",
                "lang": item.get("lang") or "ko",
                "chunk_index": item["chunk_index"],
            }
            for item in batch
        ]

        embs = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=encode_batch,
            show_progress_bar=False,
            prompt_name=PASSAGE_PROMPT,
        )
        col.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)

    print(
        f"[done] indexed {total} chunks to '{persist_dir}'/{collection} using {emb_model}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create a Chroma store from chunks.jsonl")
    ap.add_argument("--chunks", default="data_curated/chunks.jsonl")
    ap.add_argument("--persist", default="index/chroma")
    ap.add_argument("--collection", default="nmK")
    ap.add_argument("--model", default=DEF_EMB)
    ap.add_argument("--chunk-batch", type=int, default=256, help="chunks per upsert")
    ap.add_argument("--encode-batch", type=int, default=64, help="batch size for embedding model")
    ap.add_argument("--no-overwrite", action="store_true", help="append to existing collection")
    args = ap.parse_args()

    main(
        chunks_jsonl=args.chunks,
        persist_dir=args.persist,
        collection=args.collection,
        emb_model=args.model,
        chunk_batch=args.chunk_batch,
        encode_batch=args.encode_batch,
        overwrite=not args.no_overwrite,
    )
