from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass
from src.retriever import HybridRetriever
from src.llm import chat

from datetime import datetime

CURRENT_DATE = datetime.now().strftime("%Y년 %m월 %d일")
CURRENT_YEAR = datetime.now().year

SYSTEM_PROMPT = (
    f"너는 국립중앙박물관 전문 안내 도슨트이다. 친절하고 전문적으로 관람객을 도와준다. "
    f"**오늘 날짜는 {CURRENT_DATE}이다.** "
    "반드시 제공된 컨텍스트 인용문만을 근거로 답하며, 다음 역할을 수행한다:\n"

    "**시간 맥락 우선 원칙:**\n"
    f"- 현재는 {CURRENT_DATE}이므로, 이 날짜 이전에 종료된 전시나 프로그램은 '지난 전시'로 구분\n"
    "- '현재', '이번달', '지금' 등의 질문에는 현재 날짜 기준으로 진행 중인 정보만 제공\n"
    "- 관람객이 '입장료', '운영시간' 등 현재 정보를 묻는 경우, 현재 유효한 정보를 우선 제공\n"
    "- 지난 전시의 입장료나 종료된 프로그램 정보는 현재 도움이 되지 않으므로 언급하지 않음\n"
    "- 컨텍스트에 현재 진행 중인 정보가 없다면 '현재 진행 중인 특별전이 없습니다'라고 명시\n"

    "**전시 안내:**\n"
    "- 현재/예정/지난 전시를 명확히 구분하여 안내\n"
    "- 전시 기간을 먼저 확인하고 현재 진행 중인 것만 추천\n"
    "- 연령대나 관심사에 따른 맞춤 추천\n"

    "**관람 안내:**\n"
    "- 현재 유효한 운영시간, 입장료, 교통편 정보만 제공\n"
    "- 층별 구성, 관람 동선, 소요시간 안내\n"
    "- 현재 적용되는 휴무일, 특별 개관일 정보\n"

    "**교육적 해설:**\n"
    "- 유물의 역사적 배경과 문화적 의미 설명\n"
    "- 시대별, 지역별 특징 비교 분석\n"
    "- 어린이도 이해하기 쉬운 친근한 표현 사용\n"

    "**편의 서비스:**\n"
    "- 현재 이용 가능한 접근성, 편의시설, 카페/기념품점 정보\n"
    "- 현재 신청 가능한 단체 관람, 해설 프로그램 안내\n"
    "- 주차, 대중교통 이용법\n"

    "시간·요금·위치 등은 원문 그대로 사용하고, 현재 유효하지 않은 정보는 제외한다."
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

    def _filter_outdated_content(self, query: str, items: List) -> List:
        """시간에 민감한 질문에 대해 과거 정보 필터링"""
        import re
        from datetime import datetime

        # 현재 정보를 원하는 질문 패턴들
        current_info_keywords = [
            "입장료", "관람료", "요금", "티켓", "가격",
            "운영시간", "개관시간", "관람시간",
            "현재", "지금", "오늘", "이번", "이번달",
            "추천", "볼만한", "특별전"
        ]

        past_indicators = ["지난", "과거", "이전", "종료", "끝난"]

        # 현재 정보를 원하는 질문인지 확인
        wants_current = any(keyword in query for keyword in current_info_keywords)
        wants_past = any(keyword in query for keyword in past_indicators)

        if wants_current and not wants_past:
            filtered_items = []
            current_year = datetime.now().year

            for item in items:
                title = item.meta.get("title", "").lower()
                text = item.text.lower()
                score_penalty = 1.0

                # 1. 지난/과거 키워드 확인
                if any(indicator in title or indicator in text for indicator in ["지난", "과거", "종료"]):
                    score_penalty *= 0.2

                # 2. 날짜 기반 필터링 - 과거 연도 감지
                old_years = [str(year) for year in range(2000, current_year-1)]  # 2024년 이전
                text_and_title = f"{title} {text}"

                for old_year in old_years:
                    if old_year in text_and_title:
                        # 과거 연도가 많이 언급될수록 점수 더 낮춤
                        year_mentions = text_and_title.count(old_year)
                        score_penalty *= (0.1 ** year_mentions)  # 매우 낮은 점수
                        break

                # 3. URL에서 과거 전시 감지 (exhiSpThemId 등)
                url = item.meta.get("url", "")
                if "past" in url.lower() or "지난" in url:
                    score_penalty *= 0.1

                item.score *= score_penalty
                filtered_items.append(item)

            # 점수 기준으로 재정렬
            filtered_items.sort(key=lambda x: x.score, reverse=True)
            return filtered_items

        return items

    def retrieve(self, query: str, k=6) -> List[Dict[str, Any]]:
        # 박물관 도메인 키워드 확장
        expanded_query = self._expand_query(query)

        # 더 많은 후보를 가져와서 필터링
        candidate_k = min(k * 3, 20)  # 3배수로 가져오되 최대 20개
        items = self.ret.hybrid(expanded_query, top_k=candidate_k, rerank=True)

        # 시간에 민감한 질문에 대해 과거 정보 필터링
        items = self._filter_outdated_content(query, items)

        # 최종 k개만 선택
        items = items[:k]

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
                "doctype": r.meta.get("doctype", "web"),
                "exhibition_status": r.meta.get("exhibition_status", "unknown")
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
