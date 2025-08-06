import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import json
import logging

# Try to import Ollama, fallback to HuggingFace if not available
try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Fallback LLM imports
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NyaySetu Legal Q&A System", version="1.0.0")

# CORS setup - Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model for POST requests
class QueryInput(BaseModel):
    query: str

class LLMFallback:
    """Fallback LLM implementation using HuggingFace transformers"""
    def __init__(self):
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use a lightweight model for text generation
                self.generator = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-small",
                    tokenizer="microsoft/DialoGPT-small",
                    return_full_text=False,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7
                )
                logger.info("Initialized fallback LLM with HuggingFace transformers")
            except Exception as e:
                logger.error(f"Failed to initialize transformers pipeline: {e}")
                self.generator = None
        else:
            self.generator = None

    def invoke(self, prompt):
        if self.generator:
            try:
                response = self.generator(prompt, max_length=200, num_return_sequences=1)
                return response[0]['generated_text'] if response else "I apologize, but I'm unable to process your query at the moment."
            except Exception as e:
                logger.error(f"Error in fallback LLM: {e}")
                return "I apologize, but I'm unable to process your query at the moment."
        else:
            return "I apologize, but the language model is currently unavailable."

class SimpleFallbackLLM:
    """Simple rule-based fallback when no LLM is available"""
    
    def invoke(self, prompt):
        """Provide simple responses based on keywords"""
        prompt_lower = prompt.lower()
        
        if "contract" in prompt_lower:
            return "A contract is a legally binding agreement between parties. Key elements include offer, acceptance, consideration, capacity, and legality."
        elif "property" in prompt_lower:
            return "Property law governs ownership rights in real property (land) and personal property (movable items)."
        elif "criminal" in prompt_lower or "crime" in prompt_lower:
            return "Criminal law defines crimes and punishments. Elements include actus reus (guilty act) and mens rea (guilty mind)."
        elif "tort" in prompt_lower:
            return "Tort law provides remedies for civil wrongs. Types include intentional torts, negligence, and strict liability."
        elif "constitutional" in prompt_lower:
            return "Constitutional law deals with fundamental government principles including separation of powers and individual rights."
        elif "business" in prompt_lower:
            return "Business law encompasses legal rules governing commercial transactions, including contracts, employment, and corporate governance."
        else:
            return "I can help with legal questions about contracts, property, criminal law, torts, constitutional law, and business law. Please ask a specific question."

class SimpleQAChain:
    """Simple QA chain that uses keyword search in JSON data"""
    
    def __init__(self):
        self.legal_data = self.load_legal_data()
        self.llm = SimpleFallbackLLM()
    
    def load_legal_data(self):
        """Load legal data from JSON file"""
        try:
            with open("nyaysetu_pdf_texts.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load legal data: {e}")
            return []
    
    def find_relevant_content(self, query):
        """Find relevant content using simple keyword matching"""
        query_lower = query.lower()
        relevant_docs = []
        
        for doc in self.legal_data:
            text = doc.get("text", "").lower()
            source = doc.get("source", "")
            
            # Simple keyword matching
            if any(keyword in text for keyword in query_lower.split()):
                relevant_docs.append({
                    "content": doc.get("text", "")[:500] + "...",
                    "source": source
                })
        
        return relevant_docs[:3]  # Return top 3 matches
    
    def invoke(self, input_dict):
        """Process query and return response"""
        query = input_dict.get("query", "")
        
        # Find relevant documents
        relevant_docs = self.find_relevant_content(query)
        
        if relevant_docs:
            # Create context from relevant documents
            context = "\n\n".join([doc["content"] for doc in relevant_docs])
            enhanced_prompt = f"Based on this legal information: {context}\n\nQuestion: {query}\n\nAnswer:"
            answer = self.llm.invoke(enhanced_prompt)
        else:
            # Fallback to basic response
            answer = self.llm.invoke(query)
        
        return {"result": answer}

# Global variables for RAG components
vectorstore = None
retriever = None
llm = None
qa_chain = None

def initialize_rag_system():
    """Initialize the RAG system with error handling"""
    global vectorstore, retriever, llm, qa_chain
    
    try:
        # First try to initialize with full ML stack
        try:
            # Load embeddings
            logger.info("Loading embedding model...")
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
            )
            
            # Try to load existing FAISS index
            logger.info("Loading FAISS vector store...")
            try:
                vectorstore = FAISS.load_local(
                    "nyaysetu_faiss_index", 
                    embedding_model, 
                    allow_dangerous_deserialization=True
                )
                logger.info("Successfully loaded nyaysetu_faiss_index")
            except Exception as e:
                logger.warning(f"Failed to load nyaysetu_faiss_index: {e}")
                try:
                    vectorstore = FAISS.load_local(
                        "faiss_index", 
                        embedding_model, 
                        allow_dangerous_deserialization=True
                    )
                    logger.info("Successfully loaded faiss_index")
                except Exception as e2:
                    logger.error(f"Failed to load any FAISS index: {e2}")
                    # Create a minimal vector store from JSON if available
                    vectorstore = create_vectorstore_from_json(embedding_model)
            
            # Set up retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Initialize LLM with fallback
            logger.info("Initializing LLM...")
            if OLLAMA_AVAILABLE:
                try:
                    llm = OllamaLLM(model="llama3")
                    # Test the connection
                    test_response = llm.invoke("Hello")
                    logger.info("Successfully initialized Ollama LLM")
                except Exception as e:
                    logger.warning(f"Ollama not available, using fallback: {e}")
                    llm = LLMFallback()
            else:
                logger.info("Ollama not available, using fallback LLM")
                llm = LLMFallback()
            
            # Create the RetrievalQA chain
            logger.info("Creating RetrievalQA chain...")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=False
            )
            
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Full ML stack failed, falling back to simple mode: {e}")
            # Fallback to simple text search without ML models
            return initialize_simple_search_system()
        
    except Exception as e:
        logger.error(f"Failed to initialize any system: {e}")
        return initialize_simple_search_system()

def initialize_simple_search_system():
    """Initialize a simple keyword-based search system as fallback"""
    global llm, qa_chain
    
    try:
        logger.info("Initializing simple search system...")
        
        # Use simple fallback LLM
        llm = SimpleFallbackLLM()
        
        # Create a simple QA handler
        qa_chain = SimpleQAChain()
        
        logger.info("Simple search system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize simple search system: {e}")
        return False

def create_vectorstore_from_json(embedding_model):
    """Create a vectorstore from JSON data if FAISS indices are not available"""
    logger.info("Attempting to create vectorstore from JSON data...")
    
    try:
        with open("nyaysetu_pdf_texts.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        texts = []
        metadatas = []
        
        for item in data:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
                metadatas.append({"source": item.get("source", "unknown")})
            elif isinstance(item, str):
                texts.append(item)
                metadatas.append({"source": "unknown"})
        
        if texts:
            vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
            logger.info(f"Created vectorstore from {len(texts)} text chunks")
            return vectorstore
        else:
            raise ValueError("No valid texts found in JSON data")
            
    except Exception as e:
        logger.error(f"Failed to create vectorstore from JSON: {e}")
        raise

# Initialize the RAG system on startup
@app.on_event("startup")
async def startup_event():
    success = initialize_rag_system()
    if not success:
        logger.error("Failed to initialize RAG system. Some features may not work.")

@app.get("/", response_class=HTMLResponse)
async def welcome_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NyaySetu Legal Q&A System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .feature { margin: 20px 0; padding: 15px; background: #ecf0f1; border-radius: 5px; }
            .api-section { margin-top: 30px; padding: 20px; background: #e8f4f8; border-radius: 5px; }
            .endpoint { font-family: monospace; background: #34495e; color: white; padding: 5px 10px; border-radius: 3px; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏛️ NyaySetu Legal Q&A System</h1>
                <p>Advanced RAG-powered Legal Document Analysis</p>
            </div>
            
            <div class="feature">
                <h3>📚 Features</h3>
                <ul>
                    <li>Legal document question-answering using RAG (Retrieval-Augmented Generation)</li>
                    <li>FAISS vector search for efficient document retrieval</li>
                    <li>Support for multiple LLM backends (Ollama + HuggingFace fallback)</li>
                    <li>Pre-processed legal document knowledge base</li>
                </ul>
            </div>
            
            <div class="api-section">
                <h3>🔌 API Endpoints</h3>
                <p><span class="endpoint">GET /</span> - This welcome page</p>
                <p><span class="endpoint">POST /ask</span> - Submit legal questions</p>
                <p><span class="endpoint">GET /health</span> - System health check</p>
                
                <h4>Example Usage:</h4>
                <pre style="background: #2c3e50; color: white; padding: 15px; border-radius: 5px; overflow-x: auto;">
curl -X POST "http://localhost:5000/ask" \\
     -H "Content-Type: application/json" \\
     -d '{"query": "What are the key provisions of contract law?"}'
                </pre>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <p>Ready to serve legal queries! 🚀</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.post("/ask")
async def ask_question(data: QueryInput):
    """Handle legal question queries"""
    if not qa_chain:
        raise HTTPException(
            status_code=503, 
            detail="RAG system not properly initialized. Please check system logs."
        )
    
    try:
        # Validate input
        if not data.query or not data.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        query = data.query.strip()
        logger.info(f"Processing query: {query[:100]}...")
        
        # Process the query through the RAG chain
        result = qa_chain.invoke({"query": query})
        
        # Extract the answer
        answer = result.get("result", "I couldn't generate a proper response.")
        
        logger.info("Successfully processed query")
        return JSONResponse(content={
            "answer": answer,
            "query": query,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return JSONResponse(
            content={
                "error": f"Failed to process query: {str(e)}",
                "status": "error"
            }, 
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "rag_system": qa_chain is not None,
        "vectorstore": vectorstore is not None,
        "llm_type": "ollama" if OLLAMA_AVAILABLE else "fallback",
        "components": {
            "ollama_available": OLLAMA_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "vectorstore_loaded": vectorstore is not None,
            "retriever_ready": retriever is not None,
            "qa_chain_ready": qa_chain is not None
        }
    }
    
    return JSONResponse(content=status)

# For Replit deployment - bind to port 5000 for frontend access
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting NyaySetu server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
