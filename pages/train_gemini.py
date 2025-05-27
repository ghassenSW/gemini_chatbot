import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
import cassio
from dotenv import load_dotenv
import os

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TABLE_NAME = "pdf_db"

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name="models/embedding-001", api_key=None):
        genai.configure(api_key=api_key)
        self.model = model_name

    def embed_query(self, text: str):
        response = genai.embed_content(model=self.model, content=text)
        return response["embedding"]

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = genai.embed_content(model=self.model, content=text)
            embeddings.append(response["embedding"])
        return embeddings

# Initialize embeddings once
gemini_embeddings = GeminiEmbeddings(api_key=GEMINI_API_KEY)

# Initialize Astra vector store once
astra_vector_store = Cassandra(
    embedding=gemini_embeddings,
    table_name=TABLE_NAME,
)

st.title("Gemini Chatbot Trainer")

# Split text into chunks to keep token size manageable
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

uploaded_file = st.file_uploader("Upload a PDF file to train Gemini", type=["pdf"])

if uploaded_file:
    pdfreader = PdfReader(uploaded_file)
    raw_text = ""
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    texts = text_splitter.split_text(raw_text)

    # Show how many chunks extracted
    st.write(f"Extracted {len(texts)} chunks from PDF.")

    if st.button("Add data to Gemini/Astra DB"):
        astra_vector_store.add_texts(texts)
        st.success(f"Inserted {len(texts)} text chunks into Astra vector store!")

# Optional: Input box for plain text training data
text_data = st.text_area("Or paste text data here to add:")

if st.button("Add pasted text data"):
    if text_data.strip():
        texts = text_splitter.split_text(text_data)
        astra_vector_store.add_texts(texts)
        st.success(f"Inserted {len(text_data)} texts into Astra vector store!")
    else:
        st.error("Please paste some text data first.")