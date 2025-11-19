#!/usr/bin/env python3
"""
NyaySetu Setup and Initialization Script
This script sets up the complete RAG system for legal document Q&A
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'fastapi',
        'uvicorn',
        'langchain',
        'langchain_community',
        'langchain_huggingface',
        'faiss',
        'sentence_transformers',
        'transformers',
        'pydantic'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module.replace('-', '_'))
            logger.info(f"✅ {module} is available")
        except ImportError:
            missing_modules.append(module)
            logger.warning(f"❌ {module} is missing")
    
    if missing_modules:
        logger.error(f"Missing dependencies: {', '.join(missing_modules)}")
        logger.info("Please install missing dependencies")
        return False
    
    logger.info("All required dependencies are available")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        "pdfs",
        "faiss_index", 
        "nyaysetu_faiss_index",
        "__pycache__"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def extract_texts():
    """Extract texts from PDFs and create JSON"""
    logger.info("🔄 Extracting texts from documents...")
    try:
        from extract_pdf_to_json import extract_text_from_files
        success = extract_text_from_files()
        if success:
            logger.info("✅ Text extraction completed")
            return True
        else:
            logger.error("❌ Text extraction failed")
            return False
    except Exception as e:
        logger.error(f"Error in text extraction: {e}")
        return False

def build_vector_store():
    """Build FAISS vector store"""
    logger.info("🔄 Building vector store...")
    try:
        from build_vectorstore import build_vectorstore
        success = build_vectorstore()
        if success:
            logger.info("✅ Vector store built successfully")
            return True
        else:
            logger.error("❌ Vector store building failed")
            return False
    except Exception as e:
        logger.error(f"Error building vector store: {e}")
        return False

def test_system():
    """Test the RAG system"""
    logger.info("🔄 Testing RAG system...")
    try:
        from query_vectorstore import query_vectorstore
        
        test_query = "What is contract law?"
        results = query_vectorstore(test_query, k=2)
        
        if results:
            logger.info(f"✅ System test passed - found {len(results)} relevant documents")
            return True
        else:
            logger.warning("⚠️  System test found no results")
            return False
            
    except Exception as e:
        logger.error(f"Error testing system: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    logger.info("🚀 Starting NyaySetu server...")
    try:
        port = int(os.environ.get("PORT", 5000))
        logger.info(f"Server will start on port {port}")
        
        # Import and run the app
        from app import app
        import uvicorn
        
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
        
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return False

def main():
    """Main setup function"""
    print("🏛️  NyayaDwar Legal RAG System Setup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Please install required dependencies first")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Extract texts
    if not extract_texts():
        logger.error("Failed to extract texts")
        sys.exit(1)
    
    # Build vector store
    if not build_vector_store():
        logger.error("Failed to build vector store")
        sys.exit(1)
    
    # Test system
    if not test_system():
        logger.warning("System test had issues, but continuing...")
    
    # Start server
    logger.info("✅ Setup completed successfully!")
    logger.info("🚀 Starting server...")
    start_server()

if __name__ == "__main__":
    main()
