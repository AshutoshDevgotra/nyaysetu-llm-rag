import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Ollama, fallback to transformers if not available
try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available, will use fallback LLM")

# Fallback imports
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class LocalRAGSystem:
    """Local RAG system with Llama3 and fallback options"""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize all components of the RAG system"""
        try:
            # Load embeddings
            logger.info("Loading embedding model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
            )
            
            # Load vectorstore
            logger.info("Loading vectorstore...")
            try:
                self.vectorstore = FAISS.load_local(
                    "nyaysetu_faiss_index", 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded nyaysetu_faiss_index")
            except Exception as e:
                logger.warning(f"Failed to load nyaysetu_faiss_index: {e}")
                self.vectorstore = FAISS.load_local(
                    "faiss_index", 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded faiss_index")
            
            # Initialize LLM
            self.initialize_llm()
            
            # Create retrieval QA chain
            if self.llm and self.vectorstore:
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=retriever,
                    return_source_documents=True
                )
                logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
    
    def initialize_llm(self):
        """Initialize LLM with fallback options"""
        if OLLAMA_AVAILABLE:
            try:
                logger.info("Attempting to initialize Ollama LLM...")
                self.llm = OllamaLLM(model="llama3")
                # Test the connection
                test_response = self.llm.invoke("Hello")
                logger.info("Successfully initialized Ollama LLM")
                return
            except Exception as e:
                logger.warning(f"Ollama initialization failed: {e}")
        
        # Fallback to transformers
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Initializing fallback LLM with transformers...")
                self.llm = FallbackLLM()
                logger.info("Successfully initialized fallback LLM")
            except Exception as e:
                logger.error(f"Fallback LLM initialization failed: {e}")
                self.llm = SimpleFallbackLLM()
        else:
            logger.warning("No LLM available, using simple fallback")
            self.llm = SimpleFallbackLLM()
    
    def ask_question(self, question: str):
        """Ask a question using the RAG system"""
        if not self.qa_chain:
            return {
                "answer": "RAG system not properly initialized.",
                "source_documents": []
            }
        
        try:
            logger.info(f"Processing question: {question[:100]}...")
            result = self.qa_chain.invoke({"query": question})
            
            # Format the response
            answer = result.get("result", "No answer generated")
            source_docs = result.get("source_documents", [])
            
            return {
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "source": doc.metadata.get("source", "Unknown")
                    }
                    for doc in source_docs
                ]
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "source_documents": []
            }

class FallbackLLM:
    """Fallback LLM using HuggingFace transformers"""
    
    def __init__(self):
        try:
            self.generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                tokenizer="microsoft/DialoGPT-small",
                return_full_text=False,
                max_length=512
            )
        except Exception as e:
            logger.error(f"Failed to initialize transformers pipeline: {e}")
            self.generator = None
    
    def invoke(self, prompt):
        if self.generator:
            try:
                # Format prompt for legal context
                formatted_prompt = f"Legal Question: {prompt}\nAnswer:"
                response = self.generator(formatted_prompt, max_length=200, num_return_sequences=1)
                return response[0]['generated_text'] if response else "Unable to generate response."
            except Exception as e:
                logger.error(f"Error in fallback LLM: {e}")
                return "Unable to generate response due to technical issues."
        else:
            return "Language model unavailable."

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
        else:
            return "I can help with legal questions about contracts, property, criminal law, torts, and constitutional law. Please ask a specific question."

def main():
    """Main function to demonstrate the RAG system"""
    print("🏛️ Initializing NyaySetu Local RAG System...")
    rag_system = LocalRAGSystem()
    
    if not rag_system.qa_chain:
        print("❌ Failed to initialize RAG system")
        return
    
    print("✅ RAG system ready!")
    
    # Test questions
    test_questions = [
        "What are the key elements of a valid contract?",
        "What is the difference between real and personal property?",
        "What are the elements of negligence?",
        "What are constitutional rights?",
        "What types of crimes exist in criminal law?"
    ]
    
    for question in test_questions:
        print(f"\n🔍 Question: {question}")
        print("-" * 80)
        
        result = rag_system.ask_question(question)
        
        print(f"📝 Answer: {result['answer']}")
        
        if result['source_documents']:
            print(f"\n📚 Sources:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"  {i}. {doc['source']}: {doc['content']}")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    main()
