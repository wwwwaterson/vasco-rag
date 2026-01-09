"""
Example usage script for the Vasco RAG system.

This script demonstrates how to interact with the RAG API programmatically.
"""

import requests
import json
from typing import Dict


API_BASE_URL = "http://localhost:8000"


def check_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✓ API Health: {data['status']}")
        return True
    except requests.exceptions.ConnectionError:
        print("✗ API is not running. Please start it with: python api/main.py")
        return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def ask_question(question: str) -> Dict:
    """
    Ask a question to the RAG system.
    
    Args:
        question: The question to ask
        
    Returns:
        Dict with 'answer' and 'sources'
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("✗ Request timed out. The LLM might be taking too long.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP Error: {e}")
        if e.response.status_code == 503:
            print("  The service might not be ready. Check if Ollama is running.")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    """Run example questions."""
    print("=" * 60)
    print("Vasco da Gama RAG System - Example Usage")
    print("=" * 60)
    print()
    
    # Check health
    if not check_health():
        return
    
    print()
    
    # Example questions
    questions = [
        "Quando o Vasco foi fundado?",
        "Quais são as cores do Vasco?",
        "Quantos títulos brasileiros o Vasco tem?",
        "Quem foi o maior artilheiro da história do Vasco?",  # Might not be in docs
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 60}")
        print(f"Question {i}: {question}")
        print('─' * 60)
        
        result = ask_question(question)
        
        if result:
            print(f"\n📝 Answer:")
            print(f"   {result['answer']}")
            print(f"\n📚 Sources:")
            for source in result['sources']:
                print(f"   - {source}")
        
        print()
    
    print("=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
