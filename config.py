"""Shared configuration for the Vasco RAG system."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "vasco_da_gama"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

# ChromaDB settings
COLLECTION_NAME = "vasco_knowledge"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval settings
TOP_K_RESULTS = 5

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"  # Can be changed to gemma, mistral, etc.
