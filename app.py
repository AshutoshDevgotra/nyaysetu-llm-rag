from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

# Allow requests from any origin (for development). Use specific origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # ← Replace "*" with ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vectorstore and LLM
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = OllamaLLM(model="llama3")

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# Home page
@app.get("/", response_class=HTMLResponse)
async def welcome_page():
    return """
    <html>
        <head><title>Welcome to NyaySetu</title></head>
        <body>
            <h1>Welcome to NyaySetu Legal Platform</h1>
            <p><a href='/ask'>Go to Ask Page</a></p>
        </body>
    </html>
    """

# Input form at /ask
@app.post("/ask")
async def ask_question(query: str = Form(...)):
    try:
        result = qa_chain.invoke({"query": query})
        return JSONResponse(content={"answer": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Handle POST from form or API
@app.post("/ask")
async def ask_question(query: str = Form(...)):
    try:
        result = qa_chain.invoke({"query": query})
        return JSONResponse(content={"answer": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)