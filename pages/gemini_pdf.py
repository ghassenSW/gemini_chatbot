import streamlit as st
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
from typing import List
import cassio
from dotenv import load_dotenv
import os

load_dotenv()

# === CONFIG ===
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TABLE_NAME = "pdf_db"

# === GEMINI EMBEDDING WRAPPER ===
class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name="models/embedding-001", api_key=None):
        genai.configure(api_key=api_key)
        self.model = model_name

    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(model=self.model, content=text)
        return response["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [genai.embed_content(model=self.model, content=t)["embedding"] for t in texts]

# === INIT ASTRA CONNECTION ===
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# === INIT GEMINI EMBEDDINGS ===
gemini_embeddings = GeminiEmbeddings(api_key=GEMINI_API_KEY)

# === INIT ASTRA VECTOR STORE ===
astra_vector_store = Cassandra(
    embedding=gemini_embeddings,
    table_name=TABLE_NAME,
)

# === INIT GEMINI LLM ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7,
)

# === INIT RETRIEVAL QA CHAIN ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=astra_vector_store.as_retriever()
)

# --- Streamlit app ---
st.title("Gemini + Astra PDF QA")

question = st.text_input("Ask your question about the PDFs:")

if question:
    with st.spinner("Generating answer..."):
        answer = qa_chain.invoke({"query": question})["result"].strip()
    st.markdown("**Answer:**")
    st.write(answer)
