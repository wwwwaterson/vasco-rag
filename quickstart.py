"""
Quick start script to set up and run the Vasco RAG system.

This script automates the setup process:
1. Checks if vector store exists
2. Runs ingestion if needed
3. Provides instructions for starting the API
"""

import sys
from pathlib import Path


def main():
    """Quick start guide."""
    print("=" * 70)
    print("Vasco da Gama RAG System - Quick Start")
    print("=" * 70)
    print()
    
    # Check project structure
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "vasco_da_gama"
    vectorstore_dir = project_root / "vectorstore"
    
    print("📋 Checking project setup...")
    print()
    
    # Check data directory
    if not data_dir.exists():
        print("✗ Data directory not found!")
        print(f"  Expected: {data_dir}")
        print("  Please create it and add Markdown files.")
        return
    
    md_files = list(data_dir.glob("*.md"))
    if not md_files:
        print("✗ No Markdown files found in data directory!")
        print(f"  Location: {data_dir}")
        print("  Please add .md files with Vasco da Gama knowledge.")
        return
    
    print(f"✓ Found {len(md_files)} Markdown files:")
    for md_file in md_files:
        print(f"  - {md_file.name}")
    print()
    
    # Check vector store
    if not vectorstore_dir.exists() or not list(vectorstore_dir.glob("*")):
        print("⚠ Vector store not found or empty")
        print()
        print("📥 STEP 1: Run ingestion")
        print("  Command: python ingest/ingest_documents.py")
        print()
        print("  This will:")
        print("  - Read all Markdown files")
        print("  - Chunk the text")
        print("  - Generate embeddings")
        print("  - Store in ChromaDB")
        print()
    else:
        print("✓ Vector store exists")
        print()
    
    # Instructions for running API
    print("🚀 STEP 2: Start the API")
    print("  Command: python api/main.py")
    print("  Or: uvicorn api.main:app --reload")
    print()
    print("  The API will be available at:")
    print("  - http://localhost:8000")
    print("  - Docs: http://localhost:8000/docs")
    print()
    
    # Instructions for testing
    print("🧪 STEP 3: Test the system")
    print("  Option 1 - Interactive docs:")
    print("    Open http://localhost:8000/docs in your browser")
    print()
    print("  Option 2 - Example script:")
    print("    python example_usage.py")
    print()
    print("  Option 3 - curl:")
    print('    curl -X POST "http://localhost:8000/ask" \\')
    print('      -H "Content-Type: application/json" \\')
    print('      -d \'{"question": "Quando o Vasco foi fundado?"}\'')
    print()
    
    # Prerequisites
    print("📋 Prerequisites:")
    print("  ✓ Python 3.11")
    print("  ✓ Dependencies installed (pip install -r requirements.txt)")
    print("  ✓ Ollama running locally")
    print("  ✓ Ollama model pulled (e.g., ollama pull llama3)")
    print()
    
    print("=" * 70)
    print("Ready to start! Follow the steps above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
