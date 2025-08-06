"""
NyaySetu Legal RAG System
A legal document question-answering system using RAG with LangChain and FAISS
"""

__version__ = "1.0.0"
__author__ = "NyaySetu Team"
__description__ = "Legal Document Q&A System using RAG (Retrieval-Augmented Generation)"

# Make key components available at package level
try:
    from .app import app
    from .query_vectorstore import query_vectorstore
    from .build_vectorstore import build_vectorstore
    from .extract_pdf_to_json import extract_text_from_files
except ImportError:
    # Handle relative imports when running as standalone scripts
    pass
