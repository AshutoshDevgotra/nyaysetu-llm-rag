from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
import os

# Load your JSON file (ensure this is a valid JSON array of objects with "text" field)
loader = JSONLoader(
    file_path='nyaysetu_pdf_texts.json',  # Path to your JSON file
    jq_schema='.[] | .text',  # Modify to point to the 'text' field or adjust if your structure is different
    text_content=True  # ✅ Extract text content from JSON objects
)

# Load documents
docs = loader.load()

# Check if documents are loaded
if not docs:
    print("⚠️ No documents loaded. Please check your JSON file format.")
else:
    print(f"✅ {len(docs)} documents loaded.")

# Split into chunks using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# Handle case where splitting doesn't produce valid chunks
if not chunks:
    print("⚠️ No chunks created. Check the size and content of your documents.")
else:
    print(f"✅ {len(chunks)} chunks created.")

# Use a small, efficient embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Save FAISS index locally
vectorstore.save_local("faiss_index")
print("✅ FAISS index saved!")
