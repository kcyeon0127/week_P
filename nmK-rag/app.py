import os
import streamlit as st
import pandas as pd
from datetime import datetime
from src.rag_chain import RAG

st.set_page_config(
    page_title="국립중앙박물관 AI 도슨트",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 헤더
st.title("🏛️ 국립중앙박물관 AI 도슨트")
st.caption("친절한 AI 도슨트가 전시, 관람, 유물에 대해 자세히 안내해 드립니다 ✨")

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

    # 검색 옵션
    st.subheader("📊 검색 설정")
    k = st.slider("검색 결과 개수", 3, 10, 6, 1)

    # 필터링 옵션
    st.subheader("🎯 관심 분야 필터")
    exhibition_filter = st.multiselect(
        "전시 유형",
        ["현재전시", "예정전시", "지난전시", "순회전시"],
        default=[]
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
    st.markdown("- 생성: Qwen2.5 (HF) / Ollama")
    st.markdown("- 데이터: 국립중앙박물관 웹사이트")

if rerun_btn:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("초기화 완료.")

if "chat" not in st.session_state:
    st.session_state.chat = []
if "rag" not in st.session_state:
    st.session_state.rag = RAG()

# 질문 입력
st.markdown("### 💬 AI 도슨트에게 질문하기")

# 빠른 질문 버튼들
quick_questions = [
    "현재 전시 추천해주세요",
    "오늘 운영시간은 언제까지인가요?",
    "입장료가 얼마인가요?",
    "지하철로 오는 방법 알려주세요",
    "국보 1호에 대해 설명해주세요",
    "어린이 체험 프로그램이 있나요?"
]

st.markdown("**빠른 질문:**")
cols = st.columns(3)
for i, question in enumerate(quick_questions):
    with cols[i % 3]:
        if st.button(f"💡 {question}", key=f"quick_{i}"):
            st.session_state.selected_question = question

# 사용자 입력
if 'selected_question' in st.session_state:
    default_question = st.session_state.selected_question
    del st.session_state.selected_question
else:
    default_question = ""

q = st.text_input(
    "질문을 입력하거나 위의 빠른 질문 버튼을 선택하세요:",
    value=default_question,
    placeholder="예: 현재 전시 중인 특별전은 무엇인가요?"
)

ask = st.button("🚀 AI 도슨트에게 질문하기", type="primary")

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
    ctx = rag.retrieve(q, k=k)
    ans = rag.generate(q, ctx)
    st.session_state.chat.append({"role":"user","content":q})
    st.session_state.chat.append({"role":"assistant","content":ans.text, "ctx":ctx, "sources":ans.sources})

# 대화 히스토리 표시
if st.session_state.chat:
    st.markdown("### 💬 대화 기록")

for turn in st.session_state.chat:
    if turn["role"]=="user":
        with st.chat_message("user"):
            st.write(f"**질문:** {turn['content']}")
    else:
        with st.chat_message("assistant", avatar="🏛️"):
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

                    for doctype, docs in ctx_by_type.items():
                        type_name = doctype_names.get(doctype, f"📄 {doctype}")
                        st.markdown(f"**{type_name}**")
                        for i, c in enumerate(docs, start=1):
                            with st.container():
                                st.markdown(f"**[{i}] {c['title']}** (신뢰도: {c['score']:.3f})")
                                if show_doctype:
                                    st.caption(f"📁 {c.get('url', 'URL 없음')}")
                                st.text_area("", c["text"], height=100, disabled=True, key=f"ctx_{doctype}_{i}")

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
    st.markdown("---")
    st.markdown("### 📊 AI 도슨트 답변 평가")
    st.caption("마지막 답변에 대한 평가를 해주세요. 서비스 개선에 도움이 됩니다.")

    # 평가 항목들
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        s1 = st.slider("🎯 정확성", 1, 5, 4, help="정보의 정확성과 사실성")
    with col2:
        s2 = st.slider("📚 충분성", 1, 5, 4, help="답변의 완성도와 상세함")
    with col3:
        s3 = st.slider("💡 명확성", 1, 5, 4, help="이해하기 쉽고 명확한 설명")
    with col4:
        s4 = st.slider("🔗 근거성", 1, 5, 4, help="출처와 인용의 적절성")

    # 질문 유형 분류
    st.markdown("**📂 질문 유형 분류:**")
    col1, col2 = st.columns(2)
    with col1:
        primary_tags = st.multiselect(
            "주요 질문 유형",
            ["전시안내", "관람정보", "교통안내", "유물해설", "해설프로그램", "편의시설"],
            default=[]
        )
    with col2:
        secondary_tags = st.multiselect(
            "세부 분류",
            ["현재전시", "과거전시", "운영시간", "입장료", "주차", "국보", "어린이", "가족"],
            default=[]
        )

    # 만족도 및 개선사항
    satisfaction = st.radio(
        "**전반적인 만족도:**",
        ["😄 매우 만족", "🙂 만족", "😐 보통", "😕 불만족", "😞 매우 불만족"],
        horizontal=True
    )

    comment = st.text_area(
        "**개선사항이나 추가 의견:**",
        placeholder="답변에서 아쉬웠던 점이나 추가로 알고 싶은 내용을 자유롭게 적어주세요..."
    )

    if st.button("💾 평가 저장하기", type="primary"):
        log_path = "ai_docent_evaluations.csv"
        last_question = ""
        last_answer = ""
        last_sources = []

        # 마지막 Q&A 쌍 찾기
        for i in range(len(st.session_state.chat)-1, -1, -1):
            turn = st.session_state.chat[i]
            if turn["role"] == "assistant":
                last_answer = turn["content"]
                last_sources = ";".join([s.get("url","") for s in turn.get("sources",[])])
                if i > 0 and st.session_state.chat[i-1]["role"] == "user":
                    last_question = st.session_state.chat[i-1]["content"]
                break

        # 평가 레코드 생성
        rec = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": last_question,
            "answer": last_answer[:500] + "..." if len(last_answer) > 500 else last_answer,
            "sources_count": len(last_sources.split(";")) if last_sources else 0,
            "accuracy": s1,
            "sufficiency": s2,
            "clarity": s3,
            "faithfulness": s4,
            "primary_tags": ";".join(primary_tags),
            "secondary_tags": ";".join(secondary_tags),
            "satisfaction": satisfaction,
            "comment": comment,
            "search_results_count": k
        }

        # CSV 저장
        import csv, os
        file_exists = os.path.exists(log_path)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rec.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(rec)

        st.success("✅ 평가가 저장되었습니다! 소중한 피드백 감사합니다.")
        st.balloons()  # 축하 애니메이션

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888888; font-size: 0.9em;'>
    🏛️ <b>국립중앙박물관 AI 도슨트</b><br>
    교육 및 연구 목적으로 제작되었습니다.<br>
    실제 박물관 방문 시 공식 안내를 참고해 주세요.
</div>
""", unsafe_allow_html=True)
