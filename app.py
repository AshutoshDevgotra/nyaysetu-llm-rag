from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from fastapi.responses import HTMLResponse

# FastAPI app init
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def welcome_page():
    return """
    <html>
        <head>
            <title>Welcome to NyaySetu</title>
        </head>
        <body>
            <h1>Welcome to NyaySetu Legal Platform</h1>
            <p>Ask your legal questions here and get instant answers based on the Indian Constitution and laws.</p>
            <p>Use the /ask endpoint to ask questions. Example:</p>
            
        </body>
    </html>
    """

# Allow frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class QuestionRequest(BaseModel):
    question: str

# Load vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Load retriever and LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = OllamaLLM(model="llama3")

# QA Chain setup
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# API route
@app.post("/ask")
async def ask_question(req: QuestionRequest):
    try:
        result = qa_chain.invoke({"query": req.question})
        return result
    except Exception as e:
        return {"error": str(e)}
