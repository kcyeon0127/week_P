"""간소화된 RAG 시스템"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from src.retriever import HybridRetriever
from src.multitarget_llm import chat
from src.constants import (
    QUERY_EXPANSION_KEYWORDS, CURRENT_INFO_KEYWORDS, PAST_INDICATORS,
    BASE_SYSTEM_PROMPT, TARGET_PROMPTS, CURRENT_YEAR
)
import re

@dataclass
class Answer:
    text: str
    citations: List[int]
    sources: List[Dict[str, str]]

class RAG:
    def __init__(self, persist_dir="index/chroma", collection="nmK"):
        self.ret = HybridRetriever(persist_dir=persist_dir, collection=collection)

    def _expand_query(self, query: str) -> str:
        """쿼리 확장"""
        expanded = query
        for key, synonyms in QUERY_EXPANSION_KEYWORDS.items():
            if key in query:
                expanded += " " + " ".join(synonyms)
        return expanded


    def _filter_outdated_content(self, query: str, items: List) -> List:
        """과거 정보 필터링"""
        wants_current = any(keyword in query for keyword in CURRENT_INFO_KEYWORDS)
        wants_past = any(keyword in query for keyword in PAST_INDICATORS)

        if wants_current and not wants_past:
            for item in items:
                title = item.meta.get("title", "").lower()
                text = item.text.lower()

                # 과거 키워드가 있으면 점수 낮춤
                if any(indicator in title or indicator in text for indicator in PAST_INDICATORS):
                    item.score *= 0.3

                # 과거 연도가 있으면 점수 낮춤
                old_years = [str(year) for year in range(2000, CURRENT_YEAR-1)]
                for old_year in old_years:
                    if old_year in f"{title} {text}":
                        item.score *= 0.5
                        break

            items.sort(key=lambda x: x.score, reverse=True)

        return items

    def retrieve(self, query: str, k=8) -> List[Dict[str, Any]]:
        """문서 검색"""
        # 간단한 쿼리 확장만 사용
        expanded_query = self._expand_query(query)

        # 검색 실행
        items = self.ret.hybrid(expanded_query, top_k=min(k*2, 16), rerank=True)

        # 과거 정보 필터링
        items = self._filter_outdated_content(query, items)

        # 최종 결과 포맷팅
        items = items[:k]
        results = []
        for i, r in enumerate(items):
            results.append({
                "rank": i+1,
                "title": r.meta.get("title", ""),
                "url": r.meta.get("url", ""),
                "text": r.text.strip(),
                "doc_id": r.meta.get("doc_id"),
                "chunk_index": r.meta.get("chunk_index"),
                "score": r.score,
                "doctype": r.meta.get("doctype", "web"),
                "exhibition_status": r.meta.get("exhibition_status", "unknown")
            })
        return results

    def generate(self, query: str, ctx: List[Dict[str, Any]], target_type: str = "general") -> Answer:
        """답변 생성"""
        # 시스템 프롬프트 생성 (LLM이 알아서 판단)
        system_prompt = BASE_SYSTEM_PROMPT + TARGET_PROMPTS.get(target_type, TARGET_PROMPTS["general"])

        # LLM 호출 (의도 감지 없이 LLM에게 맡김)
        text = chat(system_prompt, query, ctx, target_type=target_type)

        # 출처 추출
        cites = sorted(set(int(n) for n in re.findall(r"\[([0-9]+)\]", text)))
        sources = []
        for i in cites:
            if 1 <= i <= len(ctx):
                sources.append({
                    "rank": i,
                    "title": ctx[i-1]["title"],
                    "url": ctx[i-1]["url"]
                })

        return Answer(text=text, citations=cites, sources=sources)

    def answer(self, query: str, k=8, target_type: str = "general") -> Answer:
        """질문 답변"""
        ctx = self.retrieve(query, k=k)
        return self.generate(query, ctx, target_type=target_type)