import json
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_text_data():
    """Load text data from JSON file"""
    try:
        with open("nyaysetu_pdf_texts.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} documents from JSON")
        return data
    except FileNotFoundError:
        logger.error("nyaysetu_pdf_texts.json not found")
        return []
    except Exception as e:
        logger.error(f"Error loading JSON data: {e}")
        return []

def build_vectorstore():
    """Build FAISS vectorstore from text data"""
    try:
        # Load text data
        data = load_text_data()
        if not data:
            logger.error("No data available to build vectorstore")
            return False
        
        # Initialize embeddings
        logger.info("Initializing embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
        )
        
        # Prepare texts and metadata
        texts = []
        metadatas = []
        
        # Initialize text splitter for large documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                text = item.get("text", "")
                source = item.get("source", f"document_{idx}")
            elif isinstance(item, str):
                text = item
                source = f"document_{idx}"
            else:
                logger.warning(f"Skipping invalid item at index {idx}")
                continue
            
            if text.strip():
                # Split large texts into chunks
                chunks = text_splitter.split_text(text)
                for chunk_idx, chunk in enumerate(chunks):
                    texts.append(chunk)
                    metadatas.append({
                        "source": source,
                        "chunk_id": chunk_idx,
                        "document_id": idx
                    })
        
        if not texts:
            logger.error("No valid texts found to create vectorstore")
            return False
        
        logger.info(f"Creating vectorstore with {len(texts)} text chunks...")
        
        # Create FAISS vectorstore
        vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )
        
        # Save the vectorstore
        os.makedirs("nyaysetu_faiss_index", exist_ok=True)
        vectorstore.save_local("nyaysetu_faiss_index")
        
        # Also save as backup in faiss_index directory
        os.makedirs("faiss_index", exist_ok=True)
        vectorstore.save_local("faiss_index")
        
        logger.info("Vectorstore built and saved successfully!")
        
        # Test the vectorstore
        logger.info("Testing vectorstore...")
        test_query = "What is law?"
        results = vectorstore.similarity_search(test_query, k=2)
        logger.info(f"Test query returned {len(results)} results")
        
        return True
        
    except Exception as e:
        logger.error(f"Error building vectorstore: {e}")
        return False

if __name__ == "__main__":
    success = build_vectorstore()
    if success:
        print("✅ Vectorstore built successfully!")
    else:
        print("❌ Failed to build vectorstore")
