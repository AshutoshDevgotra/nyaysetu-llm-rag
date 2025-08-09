#!/usr/bin/env python3
"""
Main entry point for NyaySetu Legal RAG System
Handles initialization and server startup for Replit deployment
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_data_files():
    """Ensure required data files exist"""
    json_file = Path("nyaysetu_pdf_texts.json")
    
    if not json_file.exists():
        logger.info("JSON data file not found, creating it...")
        try:
            from extract_pdf_to_json import extract_text_from_files
            success = extract_text_from_files()
            if not success:
                logger.error("Failed to create JSON data file")
                return False
        except Exception as e:
            logger.error(f"Error creating JSON data file: {e}")
            return False
    
    return True

def ensure_vector_store():
    """Ensure vector store exists"""
    index_paths = [
        Path("nyaysetu_faiss_index"),
        Path("faiss_index")
    ]
    
    # Check if any vector store exists
    for path in index_paths:
        if path.exists() and (path / "index.faiss").exists():
            logger.info(f"Found existing vector store: {path}")
            return True
    
    # No vector store found, create one
    logger.info("No vector store found, creating it...")
    try:
        from build_vectorstore import build_vectorstore
        success = build_vectorstore()
        if not success:
            logger.error("Failed to create vector store")
            return False
        logger.info("Vector store created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return False

def setup_environment():
    """Set up the environment for deployment"""
    # Create necessary directories
    directories = ["pdfs", "faiss_index", "nyaysetu_faiss_index"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Ensure data files exist
    if not ensure_data_files():
        logger.warning("Data files setup had issues")
    
    # Ensure vector store exists
    if not ensure_vector_store():
        logger.warning("Vector store setup had issues")
    
    return True

def main():
    """Main function for NyaySetu deployment"""
    logger.info("🏛️  Starting NyaySetu Legal RAG System")
    logger.info("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Import and start the FastAPI app
    try:
        from app import app
        import uvicorn
        
        # Fixed port for consistent access
        port = 8082
        host = "0.0.0.0"
        
        logger.info(f"🚀 Starting server on {host}:{port}")
        logger.info("📝 API Endpoints:")
        logger.info("   GET  /         - Welcome page")
        logger.info("   POST /ask      - Ask legal questions")
        logger.info("   GET  /health   - Health check")
        
        # Start the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
