"""
Query system for Vasco da Gama RAG.

This module handles:
1. Vector similarity search in ChromaDB
2. Strict prompt construction with retrieved context
3. Ollama LLM integration
4. Answer generation with source attribution
"""

from typing import List, Dict, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import json


# Configuration
VECTORSTORE_DIR = Path(__file__).parent.parent / "vectorstore"
COLLECTION_NAME = "vasco_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:1b"  # Default model, can be changed
TOP_K_RESULTS = 5  # Number of chunks to retrieve


class VascoRAG:
    """
    Production RAG system for Vasco da Gama knowledge.
    
    Enforces strict constraints:
    - Only uses retrieved context
    - Never hallucinates
    - Returns "I don't know" when context is insufficient
    """
    
    def __init__(self, ollama_model: str = OLLAMA_MODEL):
        """
        Initialize the RAG system.
        
        Args:
            ollama_model: Name of the Ollama model to use
        """
        self.ollama_model = ollama_model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Connect to existing ChromaDB
        if not VECTORSTORE_DIR.exists():
            raise FileNotFoundError(
                f"Vector store not found at {VECTORSTORE_DIR}. "
                "Please run the ingestion script first."
            )
        
        self.client = chromadb.PersistentClient(
            path=str(VECTORSTORE_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
        except Exception as e:
            raise ValueError(
                f"Collection '{COLLECTION_NAME}' not found. "
                "Please run the ingestion script first."
            ) from e
    
    def retrieve_context(self, question: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Retrieve relevant context chunks for a question.
        
        Args:
            question: User's question
            top_k: Number of top results to retrieve
            
        Returns:
            List of dicts with 'text' and 'source'
        """
        # Generate embedding for the question
        question_embedding = self.embedding_model.encode(
            question,
            convert_to_numpy=True
        )
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        contexts = []
        if results['documents'] and results['documents'][0]:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                contexts.append({
                    "text": doc,
                    "source": metadata.get("source", "unknown")
                })
        print(contexts)
        return contexts
    
    def build_prompt(self, question: str, contexts: List[Dict]) -> str:
        """
        Build a strict, structured prompt for the LLM.
        
        This prompt enforces:
        - Use only the provided context
        - No external knowledge
        - No guessing
        - Explicit "I don't know" when uncertain
        
        Args:
            question: User's question
            contexts: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        # Build context section
        context_text = "\n\n".join([
            f"[Source: {ctx['source']}]\n{ctx['text']}"
            for ctx in contexts
        ])
        
        prompt = f"""You are a helpful assistant answering questions about Clube de Regatas Vasco da Gama.

CRITICAL INSTRUCTIONS:
1. You MUST answer ONLY using the context provided below
2. The context below is AUTHORITATIVE and is the ONLY source of truth
3. DO NOT use any external knowledge or information you were trained on
4. DO NOT guess or make assumptions
5. If the answer is not explicitly present in the context, you MUST respond with "I don't know"
6. Keep your answers concise and factual
7. When you provide an answer, cite which source document it came from

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""
        
        return prompt
    
    def call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API to generate an answer.
        
        Args:
            prompt: The complete prompt with context and question
            
        Returns:
            Generated answer
            
        Raises:
            ConnectionError: If Ollama is not running
            ValueError: If the model is not available
        """
        url = f"{OLLAMA_BASE_URL}/api/generate"
        
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for factual responses
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {OLLAMA_BASE_URL}. "
                "Please ensure Ollama is running locally."
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"Model '{self.ollama_model}' not found. "
                    f"Please pull the model first: ollama pull {self.ollama_model}"
                )
            raise
    
    def ask(self, question: str) -> Dict[str, any]:
        """
        Main entry point: answer a question using RAG.
        
        Args:
            question: User's question
            
        Returns:
            Dict with 'answer' and 'sources'
        """
        # Step 1: Retrieve relevant context
        contexts = self.retrieve_context(question)
        
        if not contexts:
            return {
                "answer": "I don't know",
                "sources": []
            }
        
        # Step 2: Build strict prompt
        prompt = self.build_prompt(question, contexts)
        
        # Step 3: Generate answer
        answer = self.call_ollama(prompt)
        
        # Step 4: Extract unique sources
        sources = list(set([ctx["source"] for ctx in contexts]))
        
        return {
            "answer": answer,
            "sources": sources
        }


def main():
    """
    CLI interface for testing the RAG system.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python query.py <question>")
        print('Example: python query.py "Quando o Vasco foi fundado?"')
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    
    print(f"Question: {question}\n")
    
    rag = VascoRAG()
    result = rag.ask(question)
    
    print(f"Answer: {result['answer']}\n")
    print(f"Sources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()
