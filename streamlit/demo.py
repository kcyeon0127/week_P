# streamlit run streamlit/demo.py --server.address 0.0.0.0 --server.port 8501


import streamlit as st
import pandas as pd

st.title("간단한 스트림릿 프론트엔드")
st.write("안녕하세요! 스트림릿으로 만든 웹앱입니다.")

name = st.text_input("이름을 입력하세요")
if st.button("확인"):
    st.success(f"반갑습니다, {name}님!")
    