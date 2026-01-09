"""
Ingestion pipeline for Vasco da Gama knowledge base.

This script:
1. Reads all Markdown files from data/vasco_da_gama/
2. Chunks the text into semantically meaningful segments
3. Generates embeddings using sentence-transformers
4. Persists vectors to ChromaDB for retrieval

Run this script whenever you add or update knowledge documents.
"""

import os
import re
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "vasco_da_gama"
VECTORSTORE_DIR = Path(__file__).parent.parent / "vectorstore"
COLLECTION_NAME = "vasco_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, efficient model for local use
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 100  # Overlap to preserve context


class MarkdownChunker:
    """
    Chunks Markdown documents intelligently.
    
    Strategy:
    - Split on headers first (##, ###, etc.)
    - If sections are too large, split on paragraphs
    - Maintain chunk size limits while preserving semantic meaning
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, content: str, source: str) -> List[Dict[str, str]]:
        """
        Chunk a single document into smaller pieces.
        
        Returns list of dicts with 'text', 'source', and 'chunk_id'.
        """
        chunks = []
        
        # First, try to split by headers
        sections = self._split_by_headers(content)
        
        for section in sections:
            # If section is small enough, keep it as one chunk
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                # Split large sections by paragraphs
                chunks.extend(self._split_by_paragraphs(section))
        
        # Create chunk metadata
        result = []
        for i, chunk_text in enumerate(chunks):
            if chunk_text.strip():  # Skip empty chunks
                result.append({
                    "text": chunk_text.strip(),
                    "source": source,
                    "chunk_id": f"{source}::chunk_{i}"
                })
        
        return result
    
    def _split_by_headers(self, content: str) -> List[str]:
        """Split content by Markdown headers."""
        # Split on headers (##, ###, etc.) while keeping the header with its content
        pattern = r'(^#{1,6}\s+.+$)'
        parts = re.split(pattern, content, flags=re.MULTILINE)
        
        sections = []
        current_section = ""
        
        for part in parts:
            if re.match(r'^#{1,6}\s+', part):
                # This is a header
                if current_section:
                    sections.append(current_section)
                current_section = part + "\n"
            else:
                current_section += part
        
        if current_section:
            sections.append(current_section)
        
        return sections if sections else [content]
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split large text by paragraphs, respecting chunk size."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Start new chunk with overlap from previous
                current_chunk = current_chunk[-self.overlap:] + "\n\n" + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


class DocumentIngester:
    """
    Handles the complete ingestion pipeline.
    """
    
    def __init__(self):
        self.chunker = MarkdownChunker()
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(VECTORSTORE_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        # Reset if exists to ensure clean state on re-ingestion
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Vasco da Gama knowledge base"}
        )
    
    def load_markdown_files(self) -> List[Dict[str, str]]:
        """Load all Markdown files from the data directory."""
        documents = []
        
        if not DATA_DIR.exists():
            raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
        
        md_files = list(DATA_DIR.glob("*.md"))
        
        if not md_files:
            raise FileNotFoundError(f"No Markdown files found in {DATA_DIR}")
        
        print(f"Found {len(md_files)} Markdown files")
        
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    "content": content,
                    "source": md_file.name
                })
                print(f"  - Loaded {md_file.name}")
        
        return documents
    
    def ingest(self):
        """
        Main ingestion pipeline.
        
        1. Load documents
        2. Chunk them
        3. Generate embeddings
        4. Store in ChromaDB
        """
        print("Starting ingestion pipeline...")
        print(f"Data directory: {DATA_DIR}")
        print(f"Vector store: {VECTORSTORE_DIR}")
        print()
        
        # Load documents
        documents = self.load_markdown_files()
        print()
        
        # Chunk all documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc["content"], doc["source"])
            all_chunks.extend(chunks)
            print(f"Chunked {doc['source']}: {len(chunks)} chunks")
        
        print(f"\nTotal chunks: {len(all_chunks)}")
        print()
        
        # Generate embeddings
        print("Generating embeddings...")
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Store in ChromaDB
        print("\nStoring in vector database...")
        self.collection.add(
            ids=[chunk["chunk_id"] for chunk in all_chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[{"source": chunk["source"]} for chunk in all_chunks]
        )
        
        print(f"✓ Successfully ingested {len(all_chunks)} chunks from {len(documents)} documents")
        print(f"✓ Vector store saved to: {VECTORSTORE_DIR}")


def main():
    """Run the ingestion pipeline."""
    ingester = DocumentIngester()
    ingester.ingest()


if __name__ == "__main__":
    main()
