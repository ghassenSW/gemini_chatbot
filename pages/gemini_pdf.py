import os
os.environ["CASSANDRA_NO_USE_LIBEV"] = "1"  # Force pure-Python event loop
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

load_dotenv()

# === CONFIG ===
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TABLE_NAME = "pdf_db"
SECURE_CONNECT_BUNDLE = "secure-connect-bundle.zip"  # Ensure this is in your repo

# Debug environment variables
st.write("ASTRA_DB_APPLICATION_TOKEN:", ASTRA_DB_APPLICATION_TOKEN[:10] + "..." if ASTRA_DB_APPLICATION_TOKEN else "Not set")
st.write("ASTRA_DB_ID:", ASTRA_DB_ID if ASTRA_DB_ID else "Not set")
st.write("GEMINI_API_KEY:", GEMINI_API_KEY[:10] + "..." if GEMINI_API_KEY else "Not set")

# === GEMINI EMBEDDING WRAPPER ===
class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name="models/embedding-001", api_key=None):
        try:
            genai.configure(api_key=api_key)
            self.model = model_name
            st.write("GeminiEmbeddings initialized successfully")
        except Exception as e:
            st.error(f"Error initializing GeminiEmbeddings: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        try:
            response = genai.embed_content(model=self.model, content=text)
            return response["embedding"]
        except Exception as e:
            st.error(f"Error embedding query: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return [genai.embed_content(model=self.model, content=t)["embedding"] for t in texts]
        except Exception as e:
            st.error(f"Error embedding documents: {str(e)}")
            raise

# === INIT ASTRA CONNECTION ===
try:
    cassio.init(
        token=ASTRA_DB_APPLICATION_TOKEN,
        database_id=ASTRA_DB_ID,
        secure_connect_bundle=SECURE_CONNECT_BUNDLE
    )
    st.write("Connected to Astra DB successfully")
except Exception as e:
    st.error(f"Failed to connect to Astra DB: {str(e)}")
    st.stop()

# === INIT GEMINI EMBEDDINGS ===
try:
    gemini_embeddings = GeminiEmbeddings(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Gemini embeddings: {str(e)}")
    st.stop()

# === INIT ASTRA VECTOR STORE ===
try:
    astra_vector_store = Cassandra(
        embedding=gemini_embeddings,
        table_name=TABLE_NAME,
    )
    st.write("Initialized Astra vector store successfully")
except Exception as e:
    st.error(f"Failed to initialize Astra vector store: {str(e)}")
    st.stop()

# === INIT GEMINI LLM ===
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.7,
    )
    st.write("Initialized Gemini LLM successfully")
except Exception as e:
    st.error(f"Failed to initialize Gemini LLM: {str(e)}")
    st.stop()

# === INIT RETRIEVAL QA CHAIN ===
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=astra_vector_store.as_retriever()
    )
    st.write("Initialized QA chain successfully")
except Exception as e:
    st.error(f"Failed to initialize QA chain: {str(e)}")
    st.stop()

# --- Streamlit app ---
st.title("Gemini + Astra PDF QA")

question = st.text_input("Ask your question about the PDFs:")

if question:
    with st.spinner("Generating answer..."):
        try:
            answer = qa_chain.invoke({"query": question})["result"].strip()
            st.markdown("**Answer:**")
            st.write(answer)
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")