import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_vectorstore(query: str, k: int = 3):
    """Query the FAISS vectorstore"""
    try:
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
        )
        
        # Load vectorstore
        try:
            vectorstore = FAISS.load_local(
                "nyaysetu_faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info("Loaded nyaysetu_faiss_index")
        except Exception as e:
            logger.warning(f"Failed to load nyaysetu_faiss_index: {e}")
            vectorstore = FAISS.load_local(
                "faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info("Loaded faiss_index")
        
        # Perform similarity search
        results = vectorstore.similarity_search(query, k=k)
        
        logger.info(f"Found {len(results)} relevant documents for query: {query}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error querying vectorstore: {e}")
        return []

def query_with_scores(query: str, k: int = 3):
    """Query vectorstore and return results with similarity scores"""
    try:
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
        )
        
        # Load vectorstore
        try:
            vectorstore = FAISS.load_local(
                "nyaysetu_faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception:
            vectorstore = FAISS.load_local(
                "faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        
        # Perform similarity search with scores
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        logger.info(f"Found {len(results)} relevant documents with scores")
        
        return results
        
    except Exception as e:
        logger.error(f"Error querying vectorstore with scores: {e}")
        return []

if __name__ == "__main__":
    # Test queries
    test_queries = [
        "What are the elements of a valid contract?",
        "What is property law?",
        "What are the types of crimes?",
        "What is due process?",
        "What is negligence in tort law?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        results = query_vectorstore(query, k=2)
        
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content: {doc.page_content[:200]}...")
        
        print("\n" + "="*80)
