from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass
from src.retriever import HybridRetriever
from src.llm import chat

SYSTEM_PROMPT = (
    "너는 국립중앙박물관 전문 안내 도슨트이다. 친절하고 전문적으로 관람객을 도와준다. "
    "반드시 제공된 컨텍스트 인용문만을 근거로 답하며, 다음 역할을 수행한다:\n"

    "**전시 안내:**\n"
    "- 현재/예정/지난 전시를 구분하여 안내\n"
    "- 전시 기간, 장소, 주요 작품을 구체적으로 제시\n"
    "- 연령대나 관심사에 따른 맞춤 추천\n"

    "**관람 안내:**\n"
    "- 운영시간, 입장료, 교통편 등 실용 정보 제공\n"
    "- 층별 구성, 관람 동선, 소요시간 안내\n"
    "- 휴무일, 특별 개관일 등 일정 정보\n"

    "**교육적 해설:**\n"
    "- 유물의 역사적 배경과 문화적 의미 설명\n"
    "- 시대별, 지역별 특징 비교 분석\n"
    "- 어린이도 이해하기 쉬운 친근한 표현 사용\n"

    "**편의 서비스:**\n"
    "- 접근성, 편의시설, 카페/기념품점 정보\n"
    "- 단체 관람, 해설 프로그램 안내\n"
    "- 주차, 대중교통 이용법\n"

    "시간·요금·위치 등은 원문 그대로 사용하고, 근거 부족 시에만 추가 정보 수집을 제안한다."
)

@dataclass
class Answer:
    text: str
    citations: List[int]
    sources: List[Dict[str, str]]

class RAG:
    def __init__(self, persist_dir="index/chroma", collection="nmK"):
        self.ret = HybridRetriever(persist_dir=persist_dir, collection=collection)

    def _expand_query(self, query: str) -> str:
        """박물관 도메인 특화 질문 확장"""
        keywords_map = {
            # 전시 관련
            "전시": ["전시회", "특별전", "기획전", "상설전시", "순회전"],
            "현재": ["진행", "개최", "운영", "열리는"],
            "추천": ["볼만한", "인기", "관람", "구경"],

            # 관람 정보
            "시간": ["운영시간", "개관시간", "관람시간", "오픈시간"],
            "요금": ["입장료", "관람료", "티켓", "가격"],
            "교통": ["지하철", "버스", "주차", "오시는길", "찾아오기"],
            "편의": ["화장실", "카페", "기념품", "음식점", "휴게"],

            # 유물/작품
            "국보": ["보물", "중요문화재", "문화유산"],
            "도자기": ["청자", "백자", "분청사기", "토기"],
            "불교": ["불상", "탑", "사찰", "승려"],
            "조선": ["왕조", "궁중", "양반", "민화"],

            # 해설/교육
            "해설": ["도슨트", "가이드", "안내", "설명"],
            "어린이": ["키즈", "가족", "체험", "교육"]
        }

        expanded = query
        for key, synonyms in keywords_map.items():
            if key in query:
                expanded += " " + " ".join(synonyms)

        return expanded

    def retrieve(self, query: str, k=6) -> List[Dict[str, Any]]:
        # 박물관 도메인 키워드 확장
        expanded_query = self._expand_query(query)

        items = self.ret.hybrid(expanded_query, top_k=k, rerank=True)
        out=[]
        for i, r in enumerate(items):
            out.append({
                "rank": i+1,
                "title": r.meta.get("title",""),
                "url": r.meta.get("url",""),
                "text": r.text.strip(),
                "doc_id": r.meta.get("doc_id"),
                "chunk_index": r.meta.get("chunk_index"),
                "score": r.score,
                "doctype": r.meta.get("doctype", "web")  # 문서 유형 추가
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
