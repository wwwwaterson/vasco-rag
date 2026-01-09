"""
FastAPI application for Vasco da Gama RAG system.

Provides a REST API endpoint for asking questions about Vasco da Gama.

Endpoints:
- POST /ask: Submit a question and receive an answer with sources
- GET /health: Health check endpoint
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import logging

from api.query import VascoRAG


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic models for request/response validation
class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Question about Vasco da Gama",
        examples=["Quando o Vasco foi fundado?"]
    )


class AnswerResponse(BaseModel):
    """Response model with answer and sources."""
    answer: str = Field(
        ...,
        description="Generated answer based on retrieved context"
    )
    sources: List[str] = Field(
        ...,
        description="List of source documents used to generate the answer"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str


# Initialize FastAPI app
app = FastAPI(
    title="Vasco da Gama RAG API",
    description="Local RAG system for answering questions about Clube de Regatas Vasco da Gama",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize RAG system (singleton pattern)
rag_system = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup."""
    global rag_system
    try:
        logger.info("Initializing RAG system...")
        rag_system = VascoRAG()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Vasco da Gama RAG API",
        "version": "1.0.0",
        "endpoints": {
            "ask": "POST /ask",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the RAG system.
    """
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    return HealthResponse(
        status="healthy",
        message="RAG system is operational"
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about Vasco da Gama.
    
    This endpoint:
    1. Retrieves relevant context from the vector store
    2. Constructs a strict prompt
    3. Calls Ollama to generate an answer
    4. Returns the answer with source attribution
    
    Args:
        request: QuestionRequest with the user's question
        
    Returns:
        AnswerResponse with answer and sources
        
    Raises:
        HTTPException: If RAG system is not initialized or an error occurs
    """
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    try:
        logger.info(f"Received question: {request.question}")
        
        # Process the question through RAG
        result = rag_system.ask(request.question)
        
        logger.info(f"Generated answer with {len(result['sources'])} sources")
        
        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
        
    except ConnectionError as e:
        logger.error(f"Ollama connection error: {e}")
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your question"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
