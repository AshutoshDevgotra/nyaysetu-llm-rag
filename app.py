import os
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
# No complex chain imports needed - using simple direct approach
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")
load_dotenv()  # Also try loading default .env

# Gemini / Google Generative AI
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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
    metadata: Optional[dict] = None
    sources: Optional[List[dict]] = None

class ShortResponseLLM(LLM):
    """LLM wrapper that ensures short, concise responses (1-3 lines)"""
    
    base_llm: Any = None
    model_config = {"arbitrary_types_allowed": True}
    
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
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
            # Truncate response to ensure it's concise
            sentences = response_text.split('.')
            if len(sentences) > 3:
                response_text = '. '.join(sentences[:3]) + '.'
            return response_text[:300] + "..." if len(response_text) > 300 else response_text
        except Exception as e:
            logger.error(f"Error in short response LLM: {e}")
            return "Unable to process query at the moment."

class LLMFallback(LLM):
    """Fallback LLM implementation using HuggingFace transformers"""
    
    generator: Any = None
    model_config = {"arbitrary_types_allowed": True}
    
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
    
    model_config = {"arbitrary_types_allowed": True}
    
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
current_llm_provider = "uninitialized"
current_llm_model = "uninitialized"
initialization_error = None  # Track initialization errors


def get_llm_provider() -> str:
    """Return the configured LLM provider."""
    return os.getenv("LLM_PROVIDER", "gemini").strip().lower()


def build_primary_llm():
    """Create the primary LLM based on environment configuration."""
    provider = get_llm_provider()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024"))
    model_name = os.getenv("LLM_MODEL", "llama3")
    
    if provider == "gemini":
        if not GEMINI_AVAILABLE:
            raise RuntimeError("Gemini client not installed. Please install langchain-google-genai.")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Please add it to your environment.")
        
        model_name = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        logger.info(f"Initializing Gemini model '{model_name}'")
        base_llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=api_key,
        )
        provider_name = "gemini"
    
    else:
        logger.warning(f"Unsupported LLM_PROVIDER '{provider}', falling back to transformers pipeline.")
        base_llm = LLMFallback()
        provider_name = "fallback"
        model_name = "transformers-fallback"
    
    return base_llm, provider_name, model_name

def initialize_rag_system():
    """Initialize the RAG system with error handling"""
    global vectorstore, retriever, llm, qa_chain, current_llm_provider, current_llm_model, initialization_error
    
    try:
        # First try to initialize with full ML stack
        try:
            logger.info("Initializing Gemini embedding model (models/text-embedding-004)...")
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            # Use the Gemini-specific index
            index_name = "nyayadwaar-gemini"
            
            if not gemini_api_key:
                error_msg = "GEMINI_API_KEY not set in environment variables"
                logger.error(error_msg)
                initialization_error = error_msg
                raise RuntimeError(error_msg)
            if not pinecone_api_key:
                error_msg = "PINECONE_API_KEY not set in environment variables"
                logger.error(error_msg)
                initialization_error = error_msg
                raise RuntimeError(error_msg)
            
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=gemini_api_key
            )
            logger.info("✅ Gemini embeddings initialized")
            
            # Initialize Pinecone Vector Store
            logger.info(f"Connecting to Pinecone index '{index_name}'...")
            
            from langchain_pinecone import PineconeVectorStore
            from pinecone import Pinecone
            
            # Initialize Pinecone client
            pc = Pinecone(api_key=pinecone_api_key)
            
            # Check if index exists
            existing_indexes = [index.name for index in pc.list_indexes()]
            if index_name not in existing_indexes:
                logger.warning(f"Pinecone index '{index_name}' does not exist. Please run build_vectorstore_gemini.py first.")
                # Fallback to empty or error? Let's try to initialize anyway, it might error later if empty.
            
            vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=embedding_model,
                pinecone_api_key=pinecone_api_key
            )
            logger.info(f"Successfully connected to Pinecone index '{index_name}'")
            
            # Set up retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Initialize LLM with environment-based provider selection
            logger.info("Initializing LLM...")
            try:
                llm_instance, provider_name, model_name = build_primary_llm()
                llm = llm_instance
                global current_llm_provider, current_llm_model
                current_llm_provider = provider_name
                current_llm_model = model_name
                logger.info(f"LLM initialized using provider='{provider_name}' model='{model_name}'")
            except Exception as llm_error:
                logger.warning(f"Primary LLM initialization failed: {llm_error}. Falling back to transformers pipeline.")
                llm = ShortResponseLLM(LLMFallback())
                current_llm_provider = "fallback"
                current_llm_model = "transformers-fallback"
            
            
            # Create a simple QA function using retriever and LLM
            logger.info("Creating simple RAG chain...")
            
            # Store retriever and llm globally for use in /ask endpoint
            # No complex chain needed - we'll manually format context
            
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Full ML stack failed: {str(e)}"
            logger.warning(error_msg)
            initialization_error = error_msg
            import traceback
            traceback.print_exc()
            # Fallback to simple text search without ML models
            return initialize_simple_search_system()
        
    except Exception as e:
        logger.error(f"Failed to initialize any system: {e}")
        return initialize_simple_search_system()

def initialize_simple_search_system():
    """Initialize a simple keyword-based search system as fallback"""
    global llm, qa_chain, current_llm_provider, current_llm_model
    
    try:
        logger.info("Initializing simple search system...")
        
        # Use simple fallback LLM - no wrapper needed as it's already short
        llm = SimpleFallbackLLM()
        
        # Create a simple QA handler
        qa_chain = SimpleQAChain()
        current_llm_provider = "simple-fallback"
        current_llm_model = "keyword-search"
        
        logger.info("Simple search system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize simple search system: {e}")
        return False

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
                    <li>Support for Gemini primary backend with HuggingFace/simple fallback</li>
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
    
    # Check if RAG system is ready (retriever and llm)
    if not retriever or not llm:
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
        
        # Retrieve relevant documents
        docs = retriever.invoke(query)
        
        # Format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt with context
        prompt = f"""You are a legal assistant. Use the following context to answer the question comprehensively but concisely. Aim for an answer around 500 words. Ensure it is accurate, relevant, and clear.

Context:
{context}

Question: {query}

Answer:"""
        
        # Get answer from LLM
        answer = llm.invoke(prompt)
        
        # Extract text from response
        if hasattr(answer, 'content'):
            answer_text = answer.content
        else:
            answer_text = str(answer)
        
        # Extract sources
        sources = []
        for doc in docs:
            sources.append({
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            })
        
        logger.info(f"Successfully processed query from frontend, returning answer of length: {len(answer_text)}")
        return QueryResponse(
            answer=answer_text,
            query=query,
            status="success",
            metadata={
                "model": current_llm_model,
                "provider": current_llm_provider,
                "timestamp": "now" # You might want to use actual timestamp
            },
            sources=sources
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
        "rag_system": retriever is not None and llm is not None,
        "vectorstore": vectorstore is not None,
        "llm_type": current_llm_provider,
        "llm_model": current_llm_model,
        "components": {
            "vectorstore_loaded": vectorstore is not None,
            "retriever_ready": retriever is not None,
            "llm_ready": llm is not None
        }
    }
    return JSONResponse(content=status)

@app.get("/debug_info")
async def debug_info():
    """Return debugging info about LLM provider and vectorstore type."""
    info = {
        "llm_provider": current_llm_provider,
        "llm_model": current_llm_model,
        "vectorstore_type": "Pinecone" if vectorstore is not None else "None",
        "gemini_available": GEMINI_AVAILABLE,
        "pinecone_api_key_set": bool(os.getenv("PINECONE_API_KEY")),
        "gemini_api_key_set": bool(os.getenv("GEMINI_API_KEY")),
        "initialization_error": initialization_error,
        "retriever_ready": retriever is not None,
        "llm_ready": llm is not None
    }
    return JSONResponse(content=info)


@app.get("/test")
async def test_endpoint():
    """Simple test endpoint for frontend connectivity"""
    return JSONResponse(content={
        "message": "Backend is running successfully!",
        "port": FIXED_PORT,
        "llm_provider": current_llm_provider,
        "llm_model": current_llm_model,
        "timestamp": "2024-01-01T00:00:00Z",
        "cors_enabled": True
    })



# Fixed port configuration
FIXED_PORT = 8082

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting NyaySetu server on fixed port {FIXED_PORT}")
    uvicorn.run("app:app", host="0.0.0.0", port=FIXED_PORT, reload=False)
