import os
import streamlit as st
import pandas as pd
from datetime import datetime
from src.rag_chain import RAG

st.set_page_config(page_title="국립중앙박물관 RAG 챗봇", layout="wide")
st.title("국립중앙박물관 RAG 기반 Q/A 데모")
st.caption("오픈 LLM + Chroma + Hybrid 검색. 답변은 인용된 문서의 근거에 기반합니다. (데이터는 교육/연구 목적)")

with st.sidebar:
    st.header("검색/생성 옵션")
    k = st.slider("검색 상위 k", 3, 10, 6, 1)
    show_ctx = st.checkbox("컨텍스트 보기", value=True)
    rerun_btn = st.button("새 세션 초기화")
    st.markdown("---")
    st.write("**LLM 설정**")
    st.write("- Ollama 사용: OLLAMA_MODEL 환경변수 설정")
    st.write("- HF Transformers: HF_MODEL 환경변수 설정 (기본: Qwen2.5-1.5B-Instruct)")

if rerun_btn:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("초기화 완료.")

if "chat" not in st.session_state:
    st.session_state.chat = []
if "rag" not in st.session_state:
    st.session_state.rag = RAG()

q = st.text_input("질문을 입력하세요 (예: 오늘 관람 시간은?)", "")
ask = st.button("질문하기")

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

for turn in st.session_state.chat:
    if turn["role"]=="user":
        st.chat_message("user").write(turn["content"])
    else:
        with st.chat_message("assistant"):
            st.write(turn["content"])
            if show_ctx:
                with st.expander("🔎 사용된 컨텍스트 보기"):
                    for i, c in enumerate(turn.get("ctx", []), start=1):
                        st.markdown(f"**[{i}] {c['title']}**  \n{c.get('url','')}")
                        st.write(c["text"])
            with st.expander("📚 출처"):
                render_sources(turn.get("sources", []))

st.markdown("---")
st.subheader("정성 평가")
col1, col2, col3, col4 = st.columns(4)
with col1: s1 = st.slider("정확성", 1, 5, 4)
with col2: s2 = st.slider("충분성", 1, 5, 4)
with col3: s3 = st.slider("명확성", 1, 5, 4)
with col4: s4 = st.slider("근거성(인용)", 1, 5, 4)
tags = st.multiselect("질문 유형 태그", ["운영시간","요금","전시설명","오시는길","시설/편의","기타"], default=[])
comment = st.text_area("코멘트")

if st.button("평가 저장"):
    log_path = "evaluations.csv"
    last_answer = ""
    last_sources = []
    for t in reversed(st.session_state.chat):
        if t["role"]=="assistant":
            last_answer = t["content"]
            last_sources = ";".join([s.get("url","") for s in t.get("sources",[])])
            break
    rec = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": q,
        "answer": last_answer,
        "sources": last_sources,
        "accuracy": s1,
        "sufficiency": s2,
        "clarity": s3,
        "faithfulness": s4,
        "tags": ";".join(tags),
        "comment": comment
    }
    import csv, os
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rec.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(rec)
    st.success(f"저장됨: {log_path}")
