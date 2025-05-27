import streamlit as st

st.title("Main Navigation Page")
st.page_link("pages/gemini_pdf.py", label="💬 Chatbot Assistant")
st.page_link("pages/train_gemini.py", label="📚 Train a New Topic")
