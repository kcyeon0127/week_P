import os
import streamlit as st
import pandas as pd
from datetime import datetime
from src.rag import RAG

st.set_page_config(
    page_title="국립중앙박물관 AI 도슨트",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 헤더
st.title("🏛️ 국립중앙박물관 AI 도슨트")
current_date = datetime.now().strftime("%Y년 %m월 %d일")
st.caption(f"친절한 AI 도슨트가 전시, 관람, 유물에 대해 자세히 안내해 드립니다 ✨ (기준일: {current_date})")

# 추천 질문 섹션
st.markdown("---")
st.markdown("### 💡 이런 질문을 해보세요")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**📅 전시 정보**")
    st.markdown("• 현재 전시 추천해줘")
    st.markdown("• 가족과 함께 볼 전시는?")
    st.markdown("• 이번 달 특별전은?")

with col2:
    st.markdown("**⏰ 관람 안내**")
    st.markdown("• 오늘 관람 시간은?")
    st.markdown("• 입장료는 얼마인가요?")
    st.markdown("• 지하철로 어떻게 가나요?")

with col3:
    st.markdown("**🏺 유물 해설**")
    st.markdown("• 국보 1호가 뭔가요?")
    st.markdown("• 고려청자 특징은?")
    st.markdown("• 조선시대 도자기 설명해줘")

with col4:
    st.markdown("**👨‍🏫 해설 서비스**")
    st.markdown("• 도슨트 프로그램 있나요?")
    st.markdown("• 어린이 체험 프로그램은?")
    st.markdown("• 단체 관람 신청 방법은?")

st.markdown("---")

with st.sidebar:
    st.header("🔧 AI 도슨트 설정")

    # 모델 선택 옵션
    st.subheader("🤖 응답 모드 선택")
    target_type = st.radio(
        "응답 대상",
        ["일반 관람객", "어린이 관람객"],
        index=0,
        help="일반: 전문적이고 상세한 설명 / 어린이: 쉽고 재미있는 설명"
    )

    # 내부적으로 사용할 타겟 타입 매핑
    target_mapping = {
        "일반 관람객": "general",
        "어린이 관람객": "children"
    }
    selected_target = target_mapping[target_type]

    # 검색 옵션 (고정값)
    k = 10  # 검색 결과 개수 고정

    # 기본 설정: 현재 정보 우선
    time_priority = "🔥 현재 정보 우선"

    # 필터링 옵션
    st.subheader("🎯 관심 분야 필터")
    exhibition_filter = st.multiselect(
        "전시 유형",
        ["현재전시", "예정전시", "지난전시", "순회전시"],
        default=["현재전시", "예정전시"] if time_priority == "🔥 현재 정보 우선" else []
    )

    content_filter = st.multiselect(
        "콘텐츠 유형",
        ["전시정보", "해설안내", "이용안내", "유물정보"],
        default=[]
    )

    period_filter = st.selectbox(
        "시대별 필터",
        ["전체", "선사시대", "삼국시대", "통일신라", "고려시대", "조선시대", "근현대"],
        index=0
    )

    # 표시 옵션
    st.subheader("📋 표시 옵션")
    show_ctx = st.checkbox("검색된 원문 보기", value=True)
    show_doctype = st.checkbox("문서 유형 표시", value=True)

    # 시스템 제어
    st.subheader("⚙️ 시스템")
    rerun_btn = st.button("🔄 새 세션 시작")

    st.markdown("---")
    st.markdown("**🤖 AI 모델 정보**")
    st.markdown("- 검색: BGE-M3 (다국어)")
    st.markdown("- 생성: Qwen2.5 + LoRA 파인튜닝")
    if selected_target == "children":
        st.markdown("- 모드: 어린이 친화 📚")
    else:
        st.markdown("- 모드: 일반 관람객 🎓")
    st.markdown("- 데이터: 국립중앙박물관 웹사이트")

if rerun_btn:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("초기화 완료.")

if "chat" not in st.session_state:
    st.session_state.chat = []
if "rag" not in st.session_state:
    st.session_state.rag = RAG()

# ChatGPT 스타일 대화 표시
st.markdown("### 💬 AI 도슨트와 대화하기")

# 대화 이력 표시
for i, turn in enumerate(st.session_state.chat):
    if turn["role"] == "user":
        with st.chat_message("user"):
            st.write(turn["content"])
    else:
        with st.chat_message("assistant", avatar="🏛️"):
            # 응답 모드 표시
            response_mode = turn.get('target_type', 'general')
            mode_icon = "👶" if response_mode == "children" else "👨‍🎓"
            mode_text = "어린이 친화" if response_mode == "children" else "일반"
            st.caption(f"{mode_icon} {mode_text} 모드로 답변")

            st.write(turn["content"])

            # 검색 결과 표시 (접을 수 있게)
            if show_ctx and turn.get("ctx"):
                with st.expander("🔍 검색된 관련 문서들"):
                    ctx_by_type = {}
                    for c in turn.get("ctx", []):
                        doctype = c.get('doctype', 'web')
                        if doctype not in ctx_by_type:
                            ctx_by_type[doctype] = []
                        ctx_by_type[doctype].append(c)

                    for doctype, docs in ctx_by_type.items():
                        st.markdown(f"**📂 {doctype.upper()} 문서 ({len(docs)}개)**")
                        for j, c in enumerate(docs):
                            title = c.get("title", "(제목 없음)")
                            url = c.get("url", "")
                            st.markdown(f"- **[{j+1}]** {title}")
                            if url:
                                st.markdown(f"  🔗 {url}")

# ChatGPT 스타일 입력
q = st.chat_input("질문을 입력하세요... (예: 현재 전시 중인 특별전은?)")
ask = q is not None and q.strip() != ""

def render_sources(sources):
    if not sources:
        st.info("인용/출처가 없습니다. (근거 부족)")
    else:
        for s in sources:
            t = s.get("title","(제목 없음)")
            u = s.get("url","")
            st.markdown(f"- [#{s['rank']}] **{t}**  \n{u}")

if ask and q.strip():
    rag = st.session_state.rag

    # 시간 우선순위 설정에 따라 검색 조정
    if time_priority == "🔥 현재 정보 우선":
        # 현재 정보 우선 모드: RAG 내부 필터링 활성화
        ctx = rag.retrieve(q, k=k)
    else:
        # 모든 정보 포함 모드: 필터링 비활성화를 위해 과거 키워드 추가
        modified_q = q + " 지난"  # 필터링을 우회하기 위한 트릭
        ctx = rag.retrieve(modified_q, k=k)

    ans = rag.generate(q, ctx, target_type=selected_target)
    st.session_state.chat.append({"role":"user","content":q})
    st.session_state.chat.append({
        "role":"assistant",
        "content":ans.text,
        "ctx":ctx,
        "sources":ans.sources,
        "time_priority": time_priority,
        "target_type": selected_target
    })

# 대화 히스토리 표시
if st.session_state.chat:
    # 대화 기록은 위의 ChatGPT 스타일로 통합됨
    if turn["role"]=="user":
        with st.chat_message("user"):
            st.write(f"**질문:** {turn['content']}")
    else:
        with st.chat_message("assistant", avatar="🏛️"):
            # 응답 모드 표시
            response_mode = turn.get('target_type', 'general')
            mode_icon = "👶" if response_mode == "children" else "👨‍🎓"
            mode_text = "어린이 친화" if response_mode == "children" else "일반"
            st.caption(f"{mode_icon} {mode_text} 모드로 답변")

            st.write(f"**AI 도슨트:** {turn['content']}")

            # 문서 유형별 컨텍스트 그룹화
            if show_ctx and turn.get("ctx"):
                with st.expander("🔍 검색된 관련 문서들"):
                    ctx_by_type = {}
                    for c in turn.get("ctx", []):
                        doctype = c.get('doctype', 'web')
                        if doctype not in ctx_by_type:
                            ctx_by_type[doctype] = []
                        ctx_by_type[doctype].append(c)

                    doctype_names = {
                        'web': '🏛️ 전시정보',
                        'web-commentary': '👨‍🏫 해설안내',
                        'web-visitor': '🚇 이용안내'
                    }

                    # 현재 대화의 실제 인덱스 계산 (assistant 메시지들만 카운트)
                    assistant_count = sum(1 for t in st.session_state.chat if t["role"] == "assistant")

                    for doctype, docs in ctx_by_type.items():
                        type_name = doctype_names.get(doctype, f"📄 {doctype}")
                        st.markdown(f"**{type_name}**")
                        for i, c in enumerate(docs, start=1):
                            with st.container():
                                st.markdown(f"**[{i}] {c['title']}** (신뢰도: {c['score']:.3f})")
                                if show_doctype:
                                    st.caption(f"📁 {c.get('url', 'URL 없음')}")
                                # 더 고유한 키 생성: 타임스탬프 포함
                                import time
                                timestamp = str(int(time.time() * 1000))[-6:]  # 마지막 6자리
                                unique_key = f"ctx_{assistant_count}_{doctype}_{i}_{timestamp}_{c.get('doc_id', 'unknown')}"
                                st.text_area("문서 내용", c["text"], height=100, disabled=True, key=unique_key, label_visibility="collapsed")

            # 출처 정보 개선
            with st.expander("📚 참고한 출처들"):
                sources = turn.get("sources", [])
                if sources:
                    for s in sources:
                        st.markdown(f"**[{s['rank']}] {s.get('title', '제목 없음')}**")
                        if s.get('url'):
                            st.markdown(f"🔗 [{s['url']}]({s['url']})")
                        st.markdown("---")
                else:
                    st.info("📋 이 답변은 직접적인 문서 인용 없이 생성되었습니다.")

if st.session_state.chat:
    pass  # 채팅 이력이 있을 때의 추가 처리 (현재는 없음)

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888888; font-size: 0.9em;'>
    🏛️ <b>국립중앙박물관 AI 도슨트</b><br>
    교육 및 연구 목적으로 제작되었습니다.<br>
    실제 박물관 방문 시 공식 안내를 참고해 주세요.
</div>
""", unsafe_allow_html=True)
