from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass
from src.retriever import HybridRetriever
from src.llm import chat

SYSTEM_PROMPT = (
    "너는 국립중앙박물관 안내 챗봇이다. 반드시 제공된 컨텍스트 인용문만을 근거로 답한다. "
    "근거가 불충분하면 '근거 부족'이라고 답하고 추가 정보 수집을 제안한다. "
    "시간·요금·위치 등 숫자/고유명사는 원문 그대로 사용한다."
)

@dataclass
class Answer:
    text: str
    citations: List[int]
    sources: List[Dict[str, str]]

class RAG:
    def __init__(self, persist_dir="index/chroma", collection="nmK"):
        self.ret = HybridRetriever(persist_dir=persist_dir, collection=collection)

    def retrieve(self, query: str, k=6) -> List[Dict[str, Any]]:
        items = self.ret.hybrid(query, top_k=k, rerank=True)
        out=[]
        for i, r in enumerate(items):
            out.append({
                "rank": i+1,
                "title": r.meta.get("title",""),
                "url": r.meta.get("url",""),
                "text": r.text.strip(),
                "doc_id": r.meta.get("doc_id"),
                "chunk_index": r.meta.get("chunk_index"),
                "score": r.score
            })
        return out

    def generate(self, query: str, ctx: List[Dict[str, Any]]) -> Answer:
        # LLM 호출
        text = chat(SYSTEM_PROMPT, query, ctx)
        # (출처: [1][3]) 패턴에서 번호 추출
        import re
        cites = sorted(set(int(n) for n in re.findall(r"\[([0-9]+)\]", text)))
        sources = []
        for i in cites:
            if 1 <= i <= len(ctx):
                sources.append({"rank": i, "title": ctx[i-1]["title"], "url": ctx[i-1]["url"]})
        return Answer(text=text, citations=cites, sources=sources)

    def answer(self, query: str, k=6) -> Answer:
        ctx = self.retrieve(query, k=k)
        return self.generate(query, ctx)
