import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv(".env.local")

def test_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return

    print(f"Found API Key: {api_key[:5]}...{api_key[-5:]}")
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.2
        )
        
        print("Sending test query to Gemini...")
        response = llm.invoke("Hello, are you working?")
        print(f"Response received: {response.content}")
        
    except Exception as e:
        print(f"Error connecting to Gemini: {e}")

if __name__ == "__main__":
    test_gemini()
