from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    mode: Literal["hybrid", "vector", "keyword"] = Field(
        default="hybrid",
        description="Retrieval mode: 'hybrid', 'vector', or 'keyword'"
    )


class DocumentResult(BaseModel):
    content: str
    filename: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    mode: str
    total_results: int
    results: List[DocumentResult]


class StatusResponse(BaseModel):
    status: str
    total_chunks: int
    chroma_persisted: bool


class AskRequest(BaseModel):
    query: str = Field(..., description="User question to answer")
    k: int = Field(default=5, ge=1, le=20, description="Number of context documents to retrieve")
    mode: Literal["hybrid", "vector", "keyword"] = Field(
        default="hybrid",
        description="Retrieval mode: 'hybrid', 'vector', or 'keyword'"
    )


class SourceReference(BaseModel):
    filename: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    snippet: Optional[str] = None


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceReference]
