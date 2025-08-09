import os
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any

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

# CORS setup - Allow frontend access with explicit configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Input models for POST requests
class QueryInput(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    query: str
    status: str

class ShortResponseLLM(LLM):
    """LLM wrapper that ensures short, concise responses (1-3 lines)"""
    
    base_llm: Any = None
    
    def __init__(self, base_llm, **kwargs):
        super().__init__(**kwargs)
        self.base_llm = base_llm
    
    @property
    def _llm_type(self) -> str:
        return "short_response"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Add instruction for short responses
        short_prompt = f"{prompt}\n\nProvide a concise answer in 1-3 sentences maximum."
        
        try:
            response = self.base_llm.invoke(short_prompt)
            # Truncate response to ensure it's concise
            sentences = response.split('.')
            if len(sentences) > 3:
                response = '. '.join(sentences[:3]) + '.'
            return response[:300] + "..." if len(response) > 300 else response
        except Exception as e:
            logger.error(f"Error in short response LLM: {e}")
            return "Unable to process query at the moment."

class LLMFallback(LLM):
    """Fallback LLM implementation using HuggingFace transformers"""
    
    generator: Any = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use a lightweight model for text generation
                self.generator = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-small",
                    tokenizer="microsoft/DialoGPT-small",
                    return_full_text=False,
                    max_length=150,  # Reduced for shorter responses
                    do_sample=True,
                    temperature=0.7
                )
                logger.info("Initialized fallback LLM with HuggingFace transformers")
            except Exception as e:
                logger.error(f"Failed to initialize transformers pipeline: {e}")
                self.generator = None
        else:
            self.generator = None
    
    @property
    def _llm_type(self) -> str:
        return "fallback"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.generator:
            try:
                response = self.generator(prompt, max_length=100, num_return_sequences=1)
                result = response[0]['generated_text'] if response else "Unable to process query."
                # Keep responses short
                return result[:200] + "..." if len(result) > 200 else result
            except Exception as e:
                logger.error(f"Error in fallback LLM: {e}")
                return "Unable to process query at the moment."
        else:
            return "Language model currently unavailable."
    
    def invoke(self, prompt):
        """Compatibility method for direct invocation"""
        return self._call(prompt)

class SimpleFallbackLLM(LLM):
    """Simple rule-based fallback when no LLM is available - provides short responses"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @property
    def _llm_type(self) -> str:
        return "simple_fallback"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Provide simple, concise responses based on keywords (1-2 sentences max)"""
        prompt_lower = prompt.lower()
        
        if "contract" in prompt_lower:
            return "A contract requires offer, acceptance, consideration, and legal capacity. It creates binding obligations between parties."
        elif "property" in prompt_lower:
            return "Property law governs ownership of real estate and personal property. It includes rights to use, transfer, and exclude others."
        elif "criminal" in prompt_lower or "crime" in prompt_lower:
            return "Criminal law defines crimes and punishments. It requires both guilty act (actus reus) and guilty mind (mens rea)."
        elif "tort" in prompt_lower:
            return "Tort law provides remedies for civil wrongs like negligence and intentional harm. It allows victims to seek compensation."
        elif "constitutional" in prompt_lower:
            return "Constitutional law governs government powers and individual rights. It ensures separation of powers and fundamental freedoms."
        elif "business" in prompt_lower or "corporate" in prompt_lower:
            return "Business law covers commercial transactions, corporate governance, and employment. It regulates how businesses operate legally."
        elif "court" in prompt_lower or "procedure" in prompt_lower:
            return "Court procedures govern how legal cases are filed and heard. They ensure fair process and proper jurisdiction."
        else:
            return "I help with legal questions on contracts, property, criminal law, torts, and constitutional matters. Please ask a specific question."
    
    def invoke(self, prompt):
        """Compatibility method for direct invocation"""
        return self._call(prompt)

class SimpleQAChain:
    """Simple QA chain that uses keyword search in JSON data - provides short answers"""
    
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
                    "content": doc.get("text", "")[:300],  # Shorter content for concise responses
                    "source": source
                })
        
        return relevant_docs[:2]  # Return top 2 matches for brevity
    
    def invoke(self, input_dict):
        """Process query and return short, concise response"""
        query = input_dict.get("query", "")
        
        # Find relevant documents
        relevant_docs = self.find_relevant_content(query)
        
        if relevant_docs:
            # Create brief context from relevant documents
            context = relevant_docs[0]["content"]  # Use only the most relevant doc
            enhanced_prompt = f"Based on this legal text: {context}\n\nQuestion: {query}\n\nProvide a brief 1-2 sentence answer:"
            answer = self.llm.invoke(enhanced_prompt)
        else:
            # Fallback to basic response
            answer = self.llm.invoke(query)
        
        # Ensure answer is concise
        sentences = answer.split('.')
        if len(sentences) > 3:
            answer = '. '.join(sentences[:3]) + '.'
        
        return {"result": answer[:250] + "..." if len(answer) > 250 else answer}

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
            
            # Initialize LLM with fallback and short response wrapper
            logger.info("Initializing LLM...")
            base_llm = None
            if OLLAMA_AVAILABLE:
                try:
                    base_llm = OllamaLLM(model="llama3")
                    # Test the connection
                    test_response = base_llm.invoke("Hello")
                    logger.info("Successfully initialized Ollama LLM")
                except Exception as e:
                    logger.warning(f"Ollama not available, using fallback: {e}")
                    base_llm = LLMFallback()
            else:
                logger.info("Ollama not available, using fallback LLM")
                base_llm = LLMFallback()
            
            # Wrap LLM to ensure short responses
            llm = ShortResponseLLM(base_llm)
            
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
        
        # Use simple fallback LLM - no wrapper needed as it's already short
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
def startup_event():
    success = initialize_rag_system()
    if not success:
        logger.error("Failed to initialize RAG system. Some features may not work.")

# Call startup event
startup_event()

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
                    <li>Short, concise responses (1-3 sentences maximum)</li>
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
curl -X POST "http://localhost:8082/ask" \\
     -H "Content-Type: application/json" \\
     -d '{"query": "What are the key provisions of contract law?"}'
                </pre>
                
                <p><strong>Note:</strong> All responses are kept short and concise (1-3 sentences) for quick understanding.</p>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <p>Ready to serve legal queries! 🚀</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.post("/ask", response_model=QueryResponse)
async def ask_question(data: QueryInput):
    """Handle legal question queries - returns short, concise answers"""
    logger.info(f"Received POST request to /ask from frontend")
    
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
        logger.info(f"Processing query from frontend: {query[:50]}...")
        
        # Process the query through the RAG chain
        result = qa_chain.invoke({"query": query})
        
        # Extract the answer and ensure it's concise
        answer = result.get("result", "Unable to generate a response.")
        
        # Double-check answer length (max 3 sentences)
        sentences = answer.split('.')
        if len(sentences) > 3:
            answer = '. '.join(sentences[:3]) + '.'
        
        # Limit to 300 characters max
        if len(answer) > 300:
            answer = answer[:297] + "..."
        
        logger.info(f"Successfully processed query from frontend, returning: {answer[:50]}...")
        return QueryResponse(
            answer=answer,
            query=query,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error processing query from frontend: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
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

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint for frontend connectivity"""
    return JSONResponse(content={
        "message": "Backend is running successfully!",
        "port": 8082,
        "timestamp": "2024-01-01T00:00:00Z",
        "cors_enabled": True
    })

@app.options("/ask")
async def ask_options():
    """Handle preflight OPTIONS request for /ask endpoint"""
    return JSONResponse(content={"message": "OK"})

# Fixed port configuration
FIXED_PORT = 8082

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting NyaySetu server on fixed port {FIXED_PORT}")
    uvicorn.run("app:app", host="0.0.0.0", port=FIXED_PORT, reload=False)
