# NyaySetu LLM RAG

A legal document question-answering system built with RAG (Retrieval-Augmented Generation) using LangChain, FAISS, and Google Gemini 1.5 Flash (with lightweight local fallbacks for offline use).

## Features

- Extract text from PDF legal documents
- Build vector store for efficient document retrieval
- FastAPI web server for question answering
- Hosted Google Gemini 1.5 Flash with automatic fallback to lightweight local responders
- CORS enabled for frontend integration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Extract PDF texts:
```bash
python extract_pdf_to_json.py
```

3. Build vector store:
```bash
python build_vectorstore.py
```

4. Configure the LLM provider via environment variables:
   ```bash
   # Use Google Gemini 1.5 Flash
   export LLM_PROVIDER=gemini
   export GEMINI_API_KEY=your_google_api_key
   export LLM_MODEL=gemini-1.5-flash
   ```
   If these variables are not provided the server falls back to the lightweight keyword-based responder.

5. Run the FastAPI server:
```bash
uvicorn app:app --reload
```

## API Endpoints

- `GET /` - Welcome page
- `POST /ask` - Submit legal questions

## Requirements

- Python 3.11+
- PDF documents in the `pdfs/` folder
- **If using Gemini**: Google Cloud account with Generative Language API enabled and a `GEMINI_API_KEY`

## LLM Configuration

The backend reads the following environment variables to decide which LLM to use:

| Variable | Description | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | `gemini` or `fallback` | `gemini` |
| `LLM_MODEL` | Model name for the provider (e.g., `gemini-1.5-flash`) | Provider specific |
| `GEMINI_API_KEY` | Google API key for Gemini access | _required when `LLM_PROVIDER=gemini`_ |
| `LLM_TEMPERATURE` | Sampling temperature | `0.2` |
| `LLM_MAX_OUTPUT_TOKENS` | Max tokens generated per response | `256` |

If no provider is configured or initialization fails, the app falls back to the lightweight keyword-based QA system.
