from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from src.retriever import HybridRetriever
from src.multitarget_llm import chat
from datetime import datetime
import re

CURRENT_DATE = datetime.now().strftime("%Y년 %m월 %d일")
CURRENT_YEAR = datetime.now().year

def get_system_prompt(target_type: str = "general") -> str:
    """타겟별 시스템 프롬프트 생성"""
    base_prompt = (
        f"너는 국립중앙박물관 전문 안내 도슨트이다. 친절하고 전문적으로 관람객을 도와준다. "
        f"**오늘 날짜는 {CURRENT_DATE}이다.** "
        "반드시 제공된 컨텍스트 인용문만을 근거로 답하며, 다음 역할을 수행한다:\n"

        "**질문 의도 파악 및 맞춤 응답:**\n"
        "- 질문자가 원하는 구체적인 정보를 정확히 파악하여 답변\n"
        "- 지하철을 물으면 지하철 정보만, 버스를 물으면 버스 정보만 제공\n"
        "- 자동차/주차를 물으면 주차장 정보만 제공\n"
        "- 컨텍스트에 질문과 관련된 정보가 없으면 '해당 정보를 찾을 수 없습니다'라고 명확히 안내\n"

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
    )

    if target_type == "children":
        base_prompt += (
            "\n**어린이 특화 안내:**\n"
            "- 어려운 용어는 쉽고 재미있게 설명\n"
            "- 어린이 눈높이에 맞춰 친근하고 이해하기 쉬운 표현 사용\n"
            "- 적절한 이모지를 사용하여 더 친숙하게 설명\n"
            "- 체험이나 놀이 요소가 있는 프로그램 우선 소개\n"
            "- 가족 단위 관람객을 위한 정보 제공\n"
        )
    else:
        base_prompt += (
            "\n**교육적 해설:**\n"
            "- 유물의 역사적 배경과 문화적 의미 설명\n"
            "- 시대별, 지역별 특징 비교 분석\n"
            "- 전문적이지만 이해하기 쉬운 설명\n"
        )

    base_prompt += (
        "\n**편의 서비스:**\n"
        "- 현재 이용 가능한 접근성, 편의시설, 카페/기념품점 정보\n"
        "- 현재 신청 가능한 단체 관람, 해설 프로그램 안내\n"
        "- 주차, 대중교통 이용법\n"

        "\n시간·요금·위치 등은 원문 그대로 사용하고, 현재 유효하지 않은 정보는 제외한다."
    )

    return base_prompt

@dataclass
class Answer:
    text: str
    citations: List[int]
    sources: List[Dict[str, str]]
    target_type: str

class ImprovedRAG:
    def __init__(self, persist_dir="index/chroma", collection="nmK"):
        self.ret = HybridRetriever(persist_dir=persist_dir, collection=collection)

    def _detect_transport_intent(self, query: str) -> Optional[str]:
        """질문에서 구체적인 교통수단 의도 파악"""
        query_lower = query.lower()

        # 지하철 관련 키워드
        subway_keywords = ["지하철", "전철", "메트로", "지하철역", "전철역", "역에서", "호선", "1호선", "2호선", "3호선", "4호선", "5호선", "6호선", "7호선", "8호선", "9호선"]
        # 버스 관련 키워드
        bus_keywords = ["버스", "시내버스", "마을버스", "공항버스", "간선버스"]
        # 자동차 관련 키워드
        car_keywords = ["자동차", "차", "주차", "주차장", "자가용", "드라이브", "승용차"]
        # 택시 관련 키워드
        taxi_keywords = ["택시", "콜택시", "카카오택시", "우버"]

        if any(keyword in query_lower for keyword in subway_keywords):
            return "subway"
        elif any(keyword in query_lower for keyword in bus_keywords):
            return "bus"
        elif any(keyword in query_lower for keyword in car_keywords):
            return "car"
        elif any(keyword in query_lower for keyword in taxi_keywords):
            return "taxi"

        return None

    def _expand_query(self, query: str) -> str:
        """박물관 도메인 특화 질문 확장 및 교통수단별 세분화"""
        # 먼저 질문의 구체적인 교통수단 의도 파악
        transport_intent = self._detect_transport_intent(query)

        keywords_map = {
            # 전시 관련
            "전시": ["전시회", "특별전", "기획전", "상설전시", "순회전"],
            "현재": ["진행", "개최", "운영", "열리는"],
            "추천": ["볼만한", "인기", "관람", "구경"],
            "어디": ["위치", "장소", "전시실", "소장", "볼 수 있는", "관람", "전시", "어디서"],
            "볼": ["관람", "구경", "관람가능", "전시", "소장", "위치"],

            # 관람 정보
            "시간": ["운영시간", "개관시간", "관람시간", "오픈시간"],
            "요금": ["입장료", "관람료", "티켓", "가격"],

            # 교통수단별 세분화된 키워드
            "지하철": ["전철", "메트로", "지하철역", "전철역", "역", "호선", "대중교통", "교통편", "가는법", "오는법"],
            "버스": ["시내버스", "마을버스", "공항버스", "대중교통", "교통편", "가는법", "오는법"],
            "자동차": ["차", "주차", "주차장", "자가용", "교통편", "가는법", "오는법"],
            "택시": ["콜택시", "카카오택시", "교통편", "가는법", "오는법"],

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

        # 교통수단 의도가 명확한 경우 해당 키워드만 확장
        if transport_intent == "subway" and "지하철" in keywords_map:
            expanded += " " + " ".join(keywords_map["지하철"]) + " 대중교통"
        elif transport_intent == "bus" and "버스" in keywords_map:
            expanded += " " + " ".join(keywords_map["버스"]) + " 대중교통"
        elif transport_intent == "car" and "자동차" in keywords_map:
            expanded += " " + " ".join(keywords_map["자동차"])
        elif transport_intent == "taxi" and "택시" in keywords_map:
            expanded += " " + " ".join(keywords_map["택시"])
        else:
            # 일반적인 키워드 확장
            for key, synonyms in keywords_map.items():
                if key in query:
                    expanded += " " + " ".join(synonyms)

        return expanded

    def _filter_by_transport_relevance(self, query: str, items: List) -> List:
        """교통수단별 관련성 기반 필터링"""
        transport_intent = self._detect_transport_intent(query)

        if not transport_intent:
            return items

        filtered_items = []

        for item in items:
            title = item.meta.get("title", "").lower()
            text = item.text.lower()
            url = item.meta.get("url", "").lower()

            relevance_score = 1.0

            if transport_intent == "subway":
                # 지하철 관련 키워드가 있으면 점수 대폭 증가
                subway_words = ["지하철", "전철", "지하철역", "전철역", "역", "호선", "메트로"]
                if any(word in title or word in text for word in subway_words):
                    relevance_score *= 5.0  # 2.0 → 5.0으로 강화
                # 자동차/주차 관련이면 점수 대폭 감소
                car_words = ["주차", "자가용", "자동차", "승용차"]
                if any(word in title or word in text for word in car_words):
                    relevance_score *= 0.1  # 0.3 → 0.1로 더 강하게 감소

            elif transport_intent == "car":
                # 주차/자동차 관련 키워드가 있으면 점수 증가
                car_words = ["주차", "자동차", "자가용", "승용차", "차량"]
                if any(word in title or word in text for word in car_words):
                    relevance_score *= 2.0
                # 지하철 관련이면 점수 감소
                subway_words = ["지하철", "전철", "역", "호선"]
                if any(word in title or word in text for word in subway_words):
                    relevance_score *= 0.3

            elif transport_intent == "bus":
                # 버스 관련 키워드가 있으면 점수 증가
                bus_words = ["버스", "시내버스", "마을버스", "공항버스"]
                if any(word in title or word in text for word in bus_words):
                    relevance_score *= 2.0
                # 다른 교통수단이면 점수 감소
                other_words = ["주차", "지하철", "전철", "역"]
                if any(word in title or word in text for word in other_words):
                    relevance_score *= 0.4

            item.score *= relevance_score
            filtered_items.append(item)

        # 점수 기준으로 재정렬
        filtered_items.sort(key=lambda x: x.score, reverse=True)
        return filtered_items

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
        # 교통수단 의도 파악
        transport_intent = self._detect_transport_intent(query)

        # 지하철 질문인 경우 강제로 교통 관련 키워드 추가
        if transport_intent == "subway":
            expanded_query = query + " 지하철 이촌역 4호선 교통 오시는길"
        else:
            # 박물관 도메인 키워드 확장 (교통수단 의도 반영)
            expanded_query = self._expand_query(query)

        # 더 많은 후보를 가져와서 필터링
        candidate_k = min(k * 3, 20)  # 3배수로 가져오되 최대 20개
        items = self.ret.hybrid(expanded_query, top_k=candidate_k, rerank=True)

        # 교통수단별 관련성 필터링
        items = self._filter_by_transport_relevance(query, items)

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

    def generate(self, query: str, ctx: List[Dict[str, Any]], target_type: str = "general") -> Answer:
        # 타겟별 시스템 프롬프트 생성
        system_prompt = get_system_prompt(target_type)

        # 질문 의도를 명확히 하는 추가 프롬프트
        transport_intent = self._detect_transport_intent(query)
        if transport_intent:
            intent_prompt = ""
            if transport_intent == "subway":
                intent_prompt = "\n\n**절대 규칙**: 질문자는 지하철 이용 방법을 묻고 있습니다. 반드시 주어진 컨텍스트만 사용하여 답변하세요. 국립중앙박물관은 이촌역(4호선)에 있습니다. 역삼역이나 다른 잘못된 역 이름을 절대 언급하지 마세요. 컨텍스트에 없는 정보는 절대 만들어내지 마세요."
            elif transport_intent == "car":
                intent_prompt = "\n\n**절대 규칙**: 질문자는 자동차/주차 정보를 묻고 있습니다. 반드시 주어진 컨텍스트만 사용하여 주차장 정보만 제공하고, 대중교통 정보는 언급하지 마세요."
            elif transport_intent == "bus":
                intent_prompt = "\n\n**절대 규칙**: 질문자는 버스 이용 방법을 묻고 있습니다. 반드시 주어진 컨텍스트만 사용하여 버스 관련 정보만 제공하세요."
            system_prompt += intent_prompt

        # LLM 호출 (target_type 전달)
        text = chat(system_prompt, query, ctx, target_type=target_type)

        # (출처: [1][3]) 패턴에서 번호 추출
        import re
        cites = sorted(set(int(n) for n in re.findall(r"\[([0-9]+)\]", text)))
        sources = []
        for i in cites:
            if 1 <= i <= len(ctx):
                sources.append({"rank": i, "title": ctx[i-1]["title"], "url": ctx[i-1]["url"]})
        return Answer(text=text, citations=cites, sources=sources, target_type=target_type)

    def answer(self, query: str, k=6, target_type: str = "general") -> Answer:
        ctx = self.retrieve(query, k=k)
        return self.generate(query, ctx, target_type=target_type)

# 기존 RAG 클래스와의 호환성을 위한 래퍼
class RAG(ImprovedRAG):
    """기존 코드 호환성을 위한 래퍼 클래스"""
    pass