import json
import os
import time
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")
load_dotenv()

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

def build_vectorstore_with_gemini():
    """Build Pinecone vectorstore using Gemini embeddings"""
    try:
        # Check for API keys
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        # Force new index name for Gemini (768 dims)
        index_name = "nyayadwaar-gemini"

        if not gemini_api_key:
            logger.error("GEMINI_API_KEY not found")
            return False
        if not pinecone_api_key:
            logger.error("PINECONE_API_KEY not found")
            return False
        
        # Load text data
        data = load_text_data()
        if not data:
            logger.error("No data available to build vectorstore")
            return False
        
        # Initialize Gemini embeddings
        logger.info("Initializing Gemini embedding model (models/text-embedding-004)...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_api_key
        )
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        if index_name not in existing_indexes:
            logger.info(f"Index '{index_name}' not found. Creating it with 768 dimensions...")
            pc.create_index(
                name=index_name,
                dimension=768, # Gemini text-embedding-004 dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            logger.info(f"Index '{index_name}' created successfully")
        else:
            logger.info(f"Index '{index_name}' already exists")

        # Prepare texts and metadata
        texts = []
        metadatas = []
        
        # Initialize text splitter
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
                continue
            
            if text.strip():
                chunks = text_splitter.split_text(text)
                for chunk_idx, chunk in enumerate(chunks):
                    texts.append(chunk)
                    metadatas.append({
                        "source": source,
                        "chunk_id": chunk_idx,
                        "document_id": idx,
                        "text": chunk
                    })
        
        if not texts:
            logger.error("No valid texts found")
            return False
        
        logger.info(f"Uploading {len(texts)} chunks to Pinecone index '{index_name}'...")
        
        PineconeVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            index_name=index_name
        )
        
        logger.info("✅ Successfully uploaded vectors to Pinecone!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error building vectorstore: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Building Pinecone Vectorstore with Gemini Embeddings")
    print("=" * 60)
    
    success = build_vectorstore_with_gemini()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SUCCESS! Data uploaded to Pinecone!")
    else:
        print("❌ FAILED to build vectorstore")
    print("=" * 60)
