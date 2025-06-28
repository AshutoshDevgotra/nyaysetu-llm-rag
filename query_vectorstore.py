from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Load vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Setup retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Load LLM
llm = OllamaLLM(model="llama3")

# Setup QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# CLI prompt
query = input("❓ Ask your legal question: ")

# Updated invoke method
result = qa_chain.invoke({"query": query})

# Print result
print("\n✅ Answer:")
print(result)
