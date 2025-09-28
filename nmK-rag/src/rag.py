"""간소화된 RAG 시스템"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from src.retriever import HybridRetriever
from src.multitarget_llm import chat
from src.constants import (
    QUERY_EXPANSION_KEYWORDS, TRANSPORT_KEYWORDS, LOCATION_KEYWORDS,
    CURRENT_INFO_KEYWORDS, PAST_INDICATORS, BASE_SYSTEM_PROMPT, TARGET_PROMPTS,
    CURRENT_YEAR
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

    def _detect_transport_intent(self, query: str) -> Optional[str]:
        """교통수단 의도 감지"""
        query_lower = query.lower()
        for transport_type, keywords in TRANSPORT_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                return transport_type
        return None

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
        # 교통수단 의도 감지
        transport_intent = self._detect_transport_intent(query)

        # 지하철 질문인 경우 강제로 교통 키워드 추가
        if transport_intent == "subway":
            expanded_query = query + " 지하철 이촌역 4호선 교통 오시는길"
        else:
            expanded_query = self._expand_query(query)

        # 검색 실행
        items = self.ret.hybrid(expanded_query, top_k=min(k*2, 16), rerank=True)

        # 교통수단별 관련성 필터링
        if transport_intent:
            for item in items:
                text_lower = item.text.lower()
                title_lower = item.meta.get("title", "").lower()

                if transport_intent == "subway":
                    subway_keywords = ["지하철", "이촌역", "4호선", "전철", "역"]
                    if any(kw in text_lower or kw in title_lower for kw in subway_keywords):
                        item.score *= 5.0  # 지하철 관련 문서 점수 높임

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
        # 시스템 프롬프트 생성
        system_prompt = BASE_SYSTEM_PROMPT + TARGET_PROMPTS.get(target_type, TARGET_PROMPTS["general"])

        # 교통수단/위치 의도 감지하여 프롬프트 추가
        transport_intent = self._detect_transport_intent(query)
        is_location_query = any(keyword in query for keyword in LOCATION_KEYWORDS)

        if transport_intent == "subway":
            system_prompt += "\n지하철 질문입니다. 이촌역 4호선 정보만 제공하세요."
        elif is_location_query:
            system_prompt += "\n위치 질문입니다. 전시실과 층수만 간단히 답변하세요."

        # LLM 호출
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