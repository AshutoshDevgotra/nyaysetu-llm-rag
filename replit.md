# NyaySetu Legal RAG System

## Overview

NyaySetu is a legal document question-answering system built using Retrieval-Augmented Generation (RAG) technology. The system combines vector search capabilities with large language models to provide accurate responses to legal queries by retrieving relevant information from a knowledge base of legal documents. The application uses FastAPI for the web interface, FAISS for vector storage, and supports multiple LLM backends including Ollama (with Llama3) and HuggingFace transformers as fallbacks.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core RAG Pipeline
The system implements a standard RAG architecture with document ingestion, vector storage, retrieval, and generation components. Legal documents are processed into chunks, embedded using sentence transformers, and stored in a FAISS vector database for efficient similarity search.

### Backend Architecture
- **FastAPI Framework**: Provides REST API endpoints with CORS support for cross-origin requests
- **Modular Design**: Separate modules for PDF extraction, vector store building, querying, and serving
- **Graceful Fallbacks**: Multiple LLM options with automatic fallback from Ollama to HuggingFace transformers

### Document Processing Pipeline
- **Text Extraction**: Supports multiple file formats with JSON storage for processed content
- **Text Chunking**: Uses RecursiveCharacterTextSplitter with 1000 character chunks and 200 character overlap
- **Embedding Generation**: Leverages sentence-transformers/paraphrase-MiniLM-L6-v2 for semantic embeddings

### Vector Storage Strategy
- **FAISS Integration**: Uses Facebook's FAISS library for efficient vector similarity search
- **Dual Index Support**: Supports both "nyaysetu_faiss_index" and "faiss_index" naming conventions
- **Persistent Storage**: Vector indices are saved to disk for reuse across sessions

### LLM Integration
- **Primary LLM**: Ollama with Llama3 model for local inference
- **Fallback System**: HuggingFace transformers pipeline when Ollama is unavailable
- **Custom LLM Wrapper**: Implements fallback class to ensure system reliability

### API Design
- **RESTful Endpoints**: Clean API structure with JSON input/output
- **Query Processing**: Accepts natural language queries and returns contextual responses
- **Error Handling**: Comprehensive logging and graceful error recovery

## External Dependencies

### Core ML Libraries
- **LangChain**: Orchestrates the RAG pipeline and provides abstractions for LLMs and vector stores
- **LangChain Community**: Provides FAISS vector store integration
- **LangChain HuggingFace**: Enables HuggingFace embeddings and model integration
- **FAISS**: Facebook's library for efficient similarity search and clustering

### Language Models
- **Ollama**: Local LLM serving platform for running Llama3 models
- **HuggingFace Transformers**: Fallback for text generation when Ollama is unavailable
- **Sentence Transformers**: For generating semantic embeddings of text chunks

### Web Framework
- **FastAPI**: Modern Python web framework for building APIs
- **Uvicorn**: ASGI server for serving FastAPI applications
- **Pydantic**: Data validation and settings management

### Development Tools
- **Python 3.11**: Runtime environment as specified in runtime.txt
- **Logging**: Built-in Python logging for system monitoring and debugging

The system is designed to be self-contained and can initialize missing components automatically, making it suitable for deployment in environments like Replit where dependencies and data may need to be set up dynamically.