# 🤖📄 PDF-QA Assistant

An intelligent PDF Question Answering app built using **Streamlit**, powered by **LangChain**, **FAISS**, and Hugging Face models. Just upload a PDF, ask any question, and get accurate, context-aware answers — instantly.

---

## 🧠 Key Features

- 📄 Upload and process PDF documents
- ✂️ Automatically chunk large texts
- 🔎 Retrieve relevant context using **vector similarity + MMR**
- 💬 Generate precise answers via **LLaMA 3.1 - 8B**
- 🖥️ Beautiful and interactive UI with Streamlit

---

## 🧰 Tech Stack & Model Overview

| Component         | Description / Model Used |
|------------------|---------------------------|
| **PDF Loader**     | `PyPDFLoader` |
| **Text Splitter**  | `RecursiveCharacterTextSplitter` |
| **Embeddings**     | `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face) |
| **Vector Store**   | `FAISS` (Facebook AI Similarity Search) |
| **Retriever**      | FAISS retriever with **MMR (Max Marginal Relevance)** |
| **LLM**            | `meta-llama/Llama-3.1-8B-Instruct` |
| **Frontend**       | `Streamlit` |
| **Environment**    | `.env` file with `HUGGINGFACEHUB_ACCESS_TOKEN` |

---

## 🐍 Python Version

This project requires **Python 3.10**.

---

## 📦 Required Libraries

```bash
streamlit
langchain
langchain-community
langchain-huggingface
huggingface-hub
python-dotenv
PyPDF
faiss-cpu
sentence-transformers
```

---

## 🛠️ Local Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Amit5620/PDF-QA-Assistant.git
cd PDF-QA-Assistant
```

### 2️⃣ Create a Virtual Environment
```bash
python3.10 -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Hugging Face API Key
```bash
touch .env
```
HUGGINGFACEHUB_ACCESS_TOKEN=your_huggingface_api_token
You can generate your token here: https://huggingface.co/settings/tokens


### 5️⃣ Run the Application
```bash
streamlit run app.py
```
Open your browser and visit: http://localhost:8501


