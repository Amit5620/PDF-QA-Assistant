# app.py
import streamlit as st
from pdf_loader import load_pdf
from text_spliting import split_text
from embedding_vector import embed_text
from retriever import get_retriever
from generation import generate_answer

st.set_page_config(page_title="PDF Q&A with RAG", layout="wide")
st.title("📄 PDF Q&A Assistant powered")

# Session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# Sidebar - Upload PDF
with st.sidebar:
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        st.success("✅ PDF uploaded successfully!")
        st.session_state.pdf_uploaded = True

        if st.button("⚙️ Process PDF"):
            with st.spinner("Processing..."):
                docs = load_pdf("uploaded.pdf")
                chunks = split_text(docs)
                vector_store = embed_text(chunks)
                retriever = get_retriever(vector_store)
                st.session_state.vector_store = vector_store
                st.session_state.retriever = retriever
            st.success("✅ PDF Processed Successfully!")

# Main Area - Question Answer Section
if st.session_state.retriever:
    st.subheader("💬 Ask Questions about your PDF")
    question = st.text_input("🔎 Enter your question")

    if question:
        with st.spinner("🤖 Thinking..."):
            answer = generate_answer(question, st.session_state.retriever)
        st.markdown("### 🧠 Answer")
        st.success(answer)
else:
    if st.session_state.pdf_uploaded:
        st.info("👉 Click 'Process PDF' in the sidebar to begin.")
    else:
        st.info("👈 Upload a PDF from the sidebar to get started.")
