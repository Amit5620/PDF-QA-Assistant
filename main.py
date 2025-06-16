from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('ml.pdf')

docs = loader.load()






splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# print(len(chunks))
# print(chunks[50])




# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# vector_store = FAISS.from_documents(chunks, embeddings)

model_name = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(model_name=model_name)

vector_store = FAISS.from_documents(chunks, embedding_model)
vector_store.save_local("faiss_vector_store")


# query = "What is machine learning?"
# results = vector_store.similarity_search(query, k=3)

# for i, result in enumerate(results):
#     print(f"\nResult {i+1}:")
#     print(result.page_content)


retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# results = retriever.invoke('What is regression')

# for i, result in enumerate(results):
#     print(f"\nResult {i+1}:")
#     print(result.page_content)




# llm = ChatHuggingFace(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="conversational",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
# )

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
# )

llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",  # Use "conversational" if required by model
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

llm = ChatHuggingFace(llm=llm_endpoint)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question = "What is regression and classification? difference between them?"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})

# print(final_prompt)








answer = llm.invoke(final_prompt)
print(answer.content)