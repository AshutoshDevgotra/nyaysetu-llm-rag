import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update to your production domain when deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model for POST requests
class QueryInput(BaseModel):
    query: str

# Load vectorstore, embeddings, retriever, and LLM
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = OllamaLLM(model="llama3")

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

@app.get("/", response_class=HTMLResponse)
async def welcome_page():
    return """
    <html>
        <head><title>Welcome to NyaySetu</title></head>
        <body>
            <h1>Welcome to NyaySetu Legal Platform</h1>
            <p><a href='/ask'>Try asking a question via frontend.</a></p>
        </body>
    </html>
    """

@app.post("/ask")
async def ask_question(data: QueryInput):
    try:
        result = qa_chain.invoke({"query": data.query})
        return JSONResponse(content={"answer": result["result"]})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Needed to make Render detect open port
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
