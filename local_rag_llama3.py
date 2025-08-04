import sys
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import argparse

# Load environment variables
load_dotenv()
import os
query = os.getenv("USER_QUERY", "Explain punishment for cheating under IPC.")

# Set up argument parser for command-line query
parser = argparse.ArgumentParser()
parser.add_argument("query", type=str, help="The query to ask")
args = parser.parse_args()

# Step 1: Load JSON documents
print(" Loading JSON documents...")
with open("nyaysetu_pdf_texts.json", "r", encoding="utf-8") as f:
    raw_docs = json.load(f)
print(f" Loaded {len(raw_docs)} documents.")

# Step 2: Convert to LangChain documents
print(" Converting to LangChain documents...")
docs = [Document(page_content=item["text"], metadata={"source": item["filename"]}) for item in raw_docs]
print(f" Created {len(docs)} LangChain documents.")

# Step 3: Split text into chunks
print(" Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
print(f" Chunked into {len(chunks)} segments.")

# Step 4: Create embeddings locally using MiniLM
print(" Embedding with MiniLM...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
print(" Embedding completed.")

# Save index for reuse
vectorstore.save_local("nyaysetu_faiss_index")
print(" FAISS index saved locally.")

# Step 5: Load LLaMA 3 (from Ollama, must be running)
print(" Loading LLaMA 3 via Ollama...")
llm = OllamaLLM(model="llama3")
print(" LLaMA 3 model loaded.")

# Step 6: Setup RAG chain
print(" Setting up RAG chain...")
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
print(" RAG chain setup complete.")

# Step 7: Ask the query passed via argument
print(" Running query...")
result = qa_chain.invoke(args.query)
print(" Query completed.")

# Step 8: Print the answer
print(" Answer:", result)
