# Vasco da Gama Local RAG System

A local Retrieval Augmented Generation system that prevents factual hallucinations by constraining LLM responses to a curated knowledge base about Clube de Regatas Vasco da Gama.

## Project Description

This system uses RAG to inject authoritative knowledge at inference time without modifying the base LLM. Markdown documents serve as the single source of truth. The LLM (gemma3:1b via Ollama) acts as a stateless text generator that operates only on retrieved context.

The architecture enforces strict constraints:
- No fine-tuning
- No model modification
- Knowledge injection only at query time
- Deterministic retrieval
- Auditable responses with source attribution

## High-Level Architecture

```
Markdown files → Chunking → Embeddings → ChromaDB
                                              ↓
User question → Embedding → Vector search → Context retrieval
                                              ↓
                            Context + Question → Ollama (gemma3:1b) → Answer + Sources
```

**Ingestion** (offline):
1. Read Markdown files from `data/vasco_da_gama/`
2. Chunk by headers and paragraphs
3. Generate embeddings with sentence-transformers
4. Persist to ChromaDB

**Query** (runtime):
1. Embed user question
2. Retrieve top-K similar chunks
3. Build strict prompt with context
4. Call Ollama for generation
5. Return answer with source attribution

## Why This Architecture

**RAG over fine-tuning**: Knowledge updates require only re-ingestion, not retraining. The base model remains unchanged. New documents can be added without touching the LLM.

**Markdown as source**: Human-readable, version-controllable, easy to audit. Non-technical contributors can update knowledge without touching code.

**Vectorstore not committed**: Generated artifacts should not be versioned. The vectorstore is reproducible from source documents. Keeps repository clean and diffs meaningful.

**Stateless LLM**: The model has no memory. Every request is independent. This ensures consistency and makes the system easier to reason about.

**Deterministic and auditable**: Low temperature, explicit prompts, and source attribution make responses predictable and verifiable.

## Folder Structure

```
vasco-rag/
├── data/
│   └── vasco_da_gama/          # Markdown knowledge base (source of truth)
├── ingest/
│   └── ingest_documents.py     # Offline ingestion pipeline
├── api/
│   ├── main.py                 # FastAPI application
│   └── query.py                # RAG query logic
├── vectorstore/                # ChromaDB storage (generated, not in git)
├── config.py                   # Shared configuration
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11
- Ollama running locally
- ~2GB disk space for model and embeddings

## Local Setup

### 1. Install Python Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Install and Configure Ollama

**Install Ollama**:
- Download from https://ollama.ai
- Follow installation instructions for your OS

**Verify installation**:
```bash
ollama --version
```

**Pull the model**:
```bash
ollama pull gemma3:1b
```

**Important**: The model name must match exactly what is configured in `api/query.py`:
```python
OLLAMA_MODEL = "gemma3:1b"
```

If you want to use a different model, update this constant and pull the corresponding model.

**Start Ollama** (if not running as a service):
```bash
ollama serve
```

### 3. Run Ingestion

This step generates the vector index from Markdown files:

```bash
python ingest/ingest_documents.py
```

Expected output:
```
Found 6 Markdown files
  - Loaded historia.md
  - Loaded identidade.md
  - Loaded lutas_sociais.md
  - Loaded titulos_oficiais.md
  - Loaded titulos_amistosos.md
  - Loaded mitos_e_correcoes.md

Total chunks: 45

Generating embeddings...
Storing in vector database...
✓ Successfully ingested 45 chunks from 6 documents
```

### 4. Run the API

```bash
python api/main.py
```

The API will be available at:
- http://localhost:8000
- Interactive docs: http://localhost:8000/docs

## Usage

### Example Request

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quando o Vasco foi fundado?"}'
```

### Example Response

```json
{
  "answer": "O Clube de Regatas Vasco da Gama foi fundado em 21 de agosto de 1898, no Rio de Janeiro, por imigrantes portugueses e descendentes.",
  "sources": ["historia.md"]
}
```

### Example: Unknown Information

If the answer is not in the knowledge base:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual é o nome do presidente atual do Vasco?"}'
```

Response:
```json
{
  "answer": "I don't know",
  "sources": []
}
```

## Model Behavior and Limitations

**Temperature**: Set to 0.1 for deterministic, factual responses. This reduces creativity but increases consistency.

**Context window**: gemma3:1b has a limited context window. The system retrieves 5 chunks (~2500 characters) to fit within this constraint.

**Prompt enforcement**: The system instructs the model to refuse answering if information is not in the context. Effectiveness depends on the model's instruction-following capability.

**Language**: The knowledge base is in Portuguese. The model handles Portuguese reasonably well but may occasionally respond in English, especially for "I don't know" cases.

**Retrieval quality**: Depends on embedding similarity. Semantically similar questions may retrieve different contexts. Chunk size and overlap are tuned for this dataset but may need adjustment for other domains.

**No conversation memory**: Each request is independent. The system does not maintain conversation history.

## Updating Knowledge

To add or modify knowledge:

1. Edit or add Markdown files in `data/vasco_da_gama/`
2. Re-run ingestion: `python ingest/ingest_documents.py`
3. Restart the API if running

The vectorstore will be rebuilt from scratch.

## Configuration

Edit `api/query.py` to change:
- `OLLAMA_MODEL`: LLM model name
- `TOP_K_RESULTS`: Number of chunks to retrieve
- `EMBEDDING_MODEL`: Sentence-transformer model

Edit `ingest/ingest_documents.py` to change:
- `CHUNK_SIZE`: Characters per chunk
- `CHUNK_OVERLAP`: Overlap between chunks

## Troubleshooting

**"Could not connect to Ollama"**: Ensure Ollama is running (`ollama serve`) and accessible at `http://localhost:11434`.

**"Model not found"**: Pull the model with `ollama pull gemma3:1b`.

**"Vector store not found"**: Run ingestion first: `python ingest/ingest_documents.py`.

**Slow responses**: The model is running locally on CPU. First request may be slower as the model loads into memory.

## License

This is a demonstration project for local RAG systems.
