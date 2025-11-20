import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables
load_dotenv(".env.local")
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_rag():
    print("="*50)
    print("🔍 RAG Debugger")
    print("="*50)

    # 1. Check Credentials
    gemini_key = os.getenv("GEMINI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = "nyayadwaar-gemini"
    
    if not gemini_key:
        print("❌ GEMINI_API_KEY missing")
        return
    if not pinecone_key:
        print("❌ PINECONE_API_KEY missing")
        return
        
    print("✅ Credentials found")

    # 2. Check Pinecone Index Stats
    try:
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"\n📊 Index Stats for '{index_name}':")
        print(f"   - Total Vectors: {stats.total_vector_count}")
        print(f"   - Dimension: {stats.dimension}")
        
        if stats.total_vector_count == 0:
            print("❌ WARNING: Index is empty! Run build_vectorstore_gemini.py")
            return
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        return

    # 3. Test Retrieval
    query = "What is the punishment for theft under IPC?"
    print(f"\n🧪 Testing Retrieval for query: '{query}'")
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_key
        )
        
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=pinecone_key
        )
        
        # Get raw results with scores
        results = vectorstore.similarity_search_with_score(query, k=3)
        
        print(f"\n📄 Retrieved {len(results)} documents:")
        for i, (doc, score) in enumerate(results):
            print(f"\n   [{i+1}] Score: {score:.4f}")
            print(f"       Source: {doc.metadata.get('source', 'unknown')}")
            print(f"       Content Preview: {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"❌ Error during retrieval: {e}")

if __name__ == "__main__":
    debug_rag()
