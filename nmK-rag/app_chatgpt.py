import os
import streamlit as st
import pandas as pd
from datetime import datetime
from src.improved_rag import RAG

st.set_page_config(
    page_title="국립중앙박물관 AI 도슨트",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 헤더
st.title("🏛️ 국립중앙박물관 AI 도슨트")
current_date = datetime.now().strftime("%Y년 %m월 %d일")
st.caption(f"친절한 AI 도슨트가 전시, 관람, 유물에 대해 자세히 안내해 드립니다 ✨ (기준일: {current_date})")

# 사이드바 설정
with st.sidebar:
    st.header("🎯 AI 도슨트 설정")

    # 응답 모드 선택
    st.subheader("👤 응답 모드")
    target_type = st.radio(
        "응답 대상 선택",
        ["일반 관람객", "어린이 관람객"],
        index=0,
        help="어린이 모드: 쉽고 재미있는 설명 + 이모지"
    )

    target_mapping = {
        "일반 관람객": "general",
        "어린이 관람객": "children"
    }
    selected_target = target_mapping[target_type]

    # 필터링 옵션
    st.subheader("🎯 관심 분야 필터")
    exhibition_filter = st.multiselect(
        "전시 유형",
        ["현재전시", "예정전시", "지난전시", "순회전시"],
        default=["현재전시", "예정전시"]
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
    st.markdown("- 검색: BAAI/BGE-M3 (다국어)")
    st.markdown("- 생성: Qwen2.5 + LoRA 파인튜닝")
    if selected_target == "children":
        st.markdown("- 모드: 어린이 친화 📚")
    else:
        st.markdown("- 모드: 일반 관람객 🎓")
    st.markdown("- 데이터: 국립중앙박물관 웹사이트")

if rerun_btn:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# 세션 상태 초기화
if "chat" not in st.session_state:
    st.session_state.chat = []
if "rag" not in st.session_state:
    st.session_state.rag = RAG()

# ChatGPT 스타일 대화 표시
for i, message in enumerate(st.session_state.chat):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="🏛️"):
            # 응답 모드 표시
            mode_icon = "📚" if message.get('target_type') == 'children' else "🎓"
            mode_text = "어린이 친화" if message.get('target_type') == 'children' else "일반"
            st.caption(f"{mode_icon} {mode_text} 모드")

            st.write(message["content"])

            # 검색 결과 표시 (접을 수 있게)
            if show_ctx and message.get("ctx"):
                with st.expander("🔍 검색된 관련 문서들"):
                    for j, doc in enumerate(message["ctx"], 1):
                        title = doc.get("title", "(제목 없음)")
                        url = doc.get("url", "")
                        doctype = doc.get("doctype", "web")

                        if show_doctype:
                            st.markdown(f"**[{j}] [{doctype.upper()}]** {title}")
                        else:
                            st.markdown(f"**[{j}]** {title}")

                        if url:
                            st.markdown(f"🔗 {url}")

                        # 내용 보기 토글
                        if st.toggle(f"내용 보기", key=f"show_{i}_{j}"):
                            st.text_area("", doc["text"], height=100, disabled=True, key=f"content_{i}_{j}", label_visibility="collapsed")

# ChatGPT 스타일 입력
if prompt := st.chat_input("질문을 입력하세요... (예: 현재 전시 중인 특별전은?)"):
    # 사용자 메시지 추가
    st.session_state.chat.append({"role": "user", "content": prompt})

    # 사용자 메시지 즉시 표시
    with st.chat_message("user"):
        st.write(prompt)

    # AI 응답 생성
    with st.chat_message("assistant", avatar="🏛️"):
        mode_icon = "📚" if selected_target == 'children' else "🎓"
        mode_text = "어린이 친화" if selected_target == 'children' else "일반"
        st.caption(f"{mode_icon} {mode_text} 모드")

        with st.spinner("AI 도슨트가 답변을 준비하고 있습니다..."):
            # 검색 및 생성
            rag = st.session_state.rag
            ctx = rag.retrieve(prompt, k=10)
            answer = rag.generate(prompt, ctx, target_type=selected_target)

            # 응답 표시
            st.write(answer)

            # 검색 결과 표시
            if show_ctx and ctx:
                with st.expander("🔍 검색된 관련 문서들"):
                    for j, doc in enumerate(ctx, 1):
                        title = doc.get("title", "(제목 없음)")
                        url = doc.get("url", "")
                        doctype = doc.get("doctype", "web")

                        if show_doctype:
                            st.markdown(f"**[{j}] [{doctype.upper()}]** {title}")
                        else:
                            st.markdown(f"**[{j}]** {title}")

                        if url:
                            st.markdown(f"🔗 {url}")

            # 어시스턴트 메시지 저장
            st.session_state.chat.append({
                "role": "assistant",
                "content": answer,
                "target_type": selected_target,
                "ctx": ctx
            })

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888888; font-size: 0.9em;'>
    🏛️ <b>국립중앙박물관 AI 도슨트</b><br>
    교육 및 연구 목적으로 제작되었습니다.<br>
    실제 박물관 방문 시 공식 안내를 참고해 주세요.
</div>
""", unsafe_allow_html=True)