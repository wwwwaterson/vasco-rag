# Vasco da Gama Local RAG System

A production-ready Retrieval Augmented Generation (RAG) system for answering questions about Clube de Regatas Vasco da Gama using local LLMs via Ollama.

## Overview

This system prevents factual hallucinations by:
- Using only authoritative Markdown documents as the knowledge source
- Injecting knowledge at inference time (no fine-tuning)
- Enforcing strict prompts that forbid external knowledge
- Running completely locally with no external API calls

## Architecture

```
┌─────────────────┐
│  Markdown Docs  │
│  (data/vasco)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Ingestion     │
│  - Chunking     │
│  - Embeddings   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ChromaDB      │
│ (vectorstore/)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  FastAPI /ask   │─────▶│    Ollama    │
│  - Retrieve     │      │   (local)    │
│  - Prompt       │◀─────│              │
│  - Generate     │      └──────────────┘
└─────────────────┘
```

## Technology Stack

- **Python 3.11**: Core language
- **FastAPI**: REST API framework
- **ChromaDB**: Persistent vector store
- **Sentence-Transformers**: Embedding generation
- **Ollama**: Local LLM inference

## Project Structure

```
vasco-rag/
├── data/
│   └── vasco_da_gama/          # Markdown knowledge documents
│       ├── historia.md
│       ├── identidade.md
│       └── titulos.md
├── ingest/
│   └── ingest_documents.py     # Ingestion pipeline
├── api/
│   ├── main.py                 # FastAPI application
│   └── query.py                # RAG query logic
├── vectorstore/                # ChromaDB storage (generated, not in git)
├── config.py                   # Shared configuration
├── requirements.txt            # Python dependencies
└── .gitignore
```

## Setup

### Prerequisites

1. **Python 3.11** installed
2. **Ollama** running locally with a model pulled:
   ```bash
   # Install Ollama from https://ollama.ai
   
   # Pull a model (choose one)
   ollama pull llama3
   ollama pull gemma
   ollama pull mistral
   ```

### Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   
   # Activate (Windows)
   .venv\Scripts\activate
   
   # Activate (Linux/Mac)
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Ingest Documents

Run this whenever you add or update Markdown files in `data/vasco_da_gama/`:

```bash
python ingest/ingest_documents.py
```

This will:
- Read all `.md` files from `data/vasco_da_gama/`
- Chunk them intelligently (by headers, then paragraphs)
- Generate embeddings using `all-MiniLM-L6-v2`
- Store vectors in `vectorstore/`

### Step 2: Start the API

```bash
# From the api/ directory
cd api
python main.py

# Or using uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Step 3: Ask Questions

**Using curl:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quando o Vasco foi fundado?"}'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "Quando o Vasco foi fundado?"}
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

**Using the interactive docs:**
1. Navigate to http://localhost:8000/docs
2. Click on `POST /ask`
3. Click "Try it out"
4. Enter your question
5. Click "Execute"

## Example Request/Response

**Request:**
```json
{
  "question": "Quais são as cores do Vasco?"
}
```

**Response:**
```json
{
  "answer": "As cores do Vasco da Gama são preto e branco, representadas na famosa faixa diagonal da camisa.",
  "sources": ["identidade.md"]
}
```

## Key Design Decisions

### 1. **Separation of Ingestion and Query**
- **Why**: Ingestion is expensive (embeddings generation). We do it once and reuse.
- **How**: Separate script (`ingest_documents.py`) that persists to ChromaDB.

### 2. **Intelligent Chunking**
- **Why**: Preserve semantic meaning while keeping chunks small enough for context windows.
- **How**: Split by Markdown headers first, then by paragraphs if needed. Maintain overlap.

### 3. **Strict Prompt Engineering**
- **Why**: Prevent hallucinations by explicitly forbidding external knowledge.
- **How**: Clear instructions in the prompt that context is authoritative and "I don't know" is acceptable.

### 4. **Low Temperature**
- **Why**: Factual responses, not creative ones.
- **How**: Set temperature to 0.1 in Ollama calls.

### 5. **Source Attribution**
- **Why**: Transparency and verification.
- **How**: Return list of source documents used in the answer.

### 6. **Persistent Vector Store**
- **Why**: Don't re-embed on every startup.
- **How**: ChromaDB with `PersistentClient` pointing to `vectorstore/` directory.

### 7. **No Fine-Tuning**
- **Why**: Per requirements, knowledge injection only at inference time.
- **How**: Use base Ollama models as-is, provide context in prompts.

## Configuration

Edit `config.py` to customize:
- `EMBEDDING_MODEL`: Change embedding model
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: Adjust chunking strategy
- `TOP_K_RESULTS`: Number of chunks to retrieve
- `OLLAMA_MODEL`: Change LLM model (must be pulled in Ollama first)

## Troubleshooting

### "Could not connect to Ollama"
- Ensure Ollama is running: `ollama serve`
- Check if the model is pulled: `ollama list`

### "Vector store not found"
- Run the ingestion script first: `python ingest/ingest_documents.py`

### "No Markdown files found"
- Ensure `.md` files exist in `data/vasco_da_gama/`

### "Model not found"
- Pull the model: `ollama pull llama3` (or your chosen model)

## Adding New Knowledge

1. Add or update `.md` files in `data/vasco_da_gama/`
2. Re-run ingestion: `python ingest/ingest_documents.py`
3. Restart the API (if running)

The vector store will be completely rebuilt with the new knowledge.

## Production Considerations

For production deployment:

1. **CORS**: Update `allow_origins` in `main.py` to specific domains
2. **Logging**: Configure proper log aggregation
3. **Monitoring**: Add health checks and metrics
4. **Rate Limiting**: Add rate limiting middleware
5. **Authentication**: Add API key or OAuth if needed
6. **Model Selection**: Test different Ollama models for accuracy/speed tradeoff
7. **Scaling**: Consider running multiple Ollama instances behind a load balancer

## License

This is a demonstration project for local RAG systems.
