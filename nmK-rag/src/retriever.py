from __future__ import annotations
import os, json
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
from dataclasses import dataclass

@dataclass
class Retrieved:
    text: str
    score: float
    meta: Dict[str, Any]

class HybridRetriever:
    def __init__(self, persist_dir="index/chroma", collection="nmK",
                 reranker:str|None="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
        self.col = self.client.get_collection(collection)
        self._docs_cache: List[str] = []
        self._metas_cache: List[Dict[str,Any]] = []
        # build BM25 corpus once
        res = self.col.get(include=["documents","metadatas","embeddings"])
        self._docs_cache = res["documents"]
        self._metas_cache = res["metadatas"]
        self.bm25 = BM25Okapi([d.split() for d in self._docs_cache])
        self.reranker = CrossEncoder(reranker) if reranker else None

    def dense_search(self, query: str, k: int = 30) -> List[Retrieved]:
        q = self.col.query(query_texts=[query], n_results=k, include=["documents","metadatas","distances"])
        out=[]
        for txt, meta, dist in zip(q["documents"][0], q["metadatas"][0], q["distances"][0]):
            out.append(Retrieved(text=txt, score=1.0 - dist, meta=meta))
        return out

    def bm25_search(self, query: str, k: int = 50) -> List[Retrieved]:
        scores = self.bm25.get_scores(query.split())
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [Retrieved(text=self._docs_cache[i], score=float(scores[i]), meta=self._metas_cache[i]) for i in idxs]

    def hybrid(self, query: str, k_dense=30, k_bm25=50, top_k=8, rerank=True) -> List[Retrieved]:
        A = self.dense_search(query, k_dense)
        B = self.bm25_search(query, k_bm25)
        pool = {(r.meta["doc_id"], r.meta["chunk_index"]): r for r in A}
        for r in B:
            key = (r.meta["doc_id"], r.meta["chunk_index"])
            if key in pool:
                pool[key].score += r.score * 0.1
            else:
                pool[key] = r
        cand = list(pool.values())
        cand = sorted(cand, key=lambda r: r.score, reverse=True)[:max(top_k*5, 30)]

        if rerank and self.reranker:
            pairs = [(query, r.text) for r in cand]
            rrs = self.reranker.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
            for r, s in zip(cand, rrs):
                r.score = float(s)
            cand = sorted(cand, key=lambda r: r.score, reverse=True)

        return cand[:top_k]
