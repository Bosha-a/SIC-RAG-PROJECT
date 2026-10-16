from typing import Optional
from fastapi import APIRouter, HTTPException

from config.utils import CHROMA_DIR
from models.schemas import (
    AskRequest,
    AskResponse,
    DocumentResult,
    QueryRequest,
    QueryResponse,
    SourceReference,
    StatusResponse,
)
from pipeline.retrieval import HybridRetrieverPipeline
from pipeline.generation import generate_answer

router = APIRouter(prefix="/api", tags=["RAG Pipeline"])

# Singleton pipeline instance
_pipeline: Optional[HybridRetrieverPipeline] = None


def get_pipeline() -> HybridRetrieverPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = HybridRetrieverPipeline()
    return _pipeline


def set_pipeline(pipeline: HybridRetrieverPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


@router.get("/status", response_model=StatusResponse)
def get_status():
    """
    Check the current status of the retrieval pipeline and database.
    """
    try:
        pipeline = get_pipeline()
        total_chunks = len(pipeline.chunks)
        chroma_exists = CHROMA_DIR.exists()
        return StatusResponse(
            status="ready",
            total_chunks=total_chunks,
            chroma_persisted=chroma_exists
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=QueryResponse)
def search_documents(request: QueryRequest):
    """
    Search documents using hybrid, vector, or keyword retrieval.
    """
    try:
        pipeline = get_pipeline()
        docs = pipeline.search(
            query=request.query,
            k=request.k,
            mode=request.mode
        )

        results = [
            DocumentResult(
                content=doc.page_content,
                filename=doc.metadata.get("filename"),
                page=doc.metadata.get("page"),
                chunk_id=str(doc.metadata.get("chunk_id", ""))
            )
            for doc in docs
        ]

        return QueryResponse(
            query=request.query,
            mode=request.mode,
            total_results=len(results),
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex")
def reindex_documents():
    """
    Force reload documents from disk and recreate vectorstore and retrievers.
    """
    try:
        global _pipeline
        from pipeline.chunking import get_chunks
        from pipeline.embedding import get_or_create_vectorstore

        new_chunks = get_chunks()
        new_vectorstore = get_or_create_vectorstore(chunks=new_chunks, force_reindex=True)
        _pipeline = HybridRetrieverPipeline(chunks=new_chunks, vectorstore=new_vectorstore)

        return {
            "status": "reindexed successfully",
            "total_chunks": len(new_chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Synthesize an answer using Google Gemini and retrieved documents with the static system prompt.
    """
    try:
        pipeline = get_pipeline()

        result = generate_answer(
            query=request.query,
            retriever=pipeline,
            k=request.k,
            mode=request.mode
        )

        sources = [
            SourceReference(
                filename=s.get("filename"),
                page=s.get("page"),
                chunk_id=str(s.get("chunk_id", "")),
                snippet=s.get("snippet")
            )
            for s in result.get("sources", [])
        ]

        return AskResponse(
            query=request.query,
            answer=result["answer"],
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

