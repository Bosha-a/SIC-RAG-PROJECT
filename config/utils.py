import os
from pathlib import Path
from dotenv import load_dotenv

# Define project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

# Directory paths
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Embedding and Model configurations
COLLECTION_NAME = "rag_documents" # vector database
JINA_API_KEY = os.getenv("JINA_API_KEY") or (
    os.getenv("EMBEDDING_MODEL") if os.getenv("EMBEDDING_MODEL", "").startswith("jina_") else None
)
JINA_MODEL_NAME = os.getenv("JINA_MODEL_NAME", "jina-embeddings-v2-base-en")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.8-flash")

# Chunking and Retrieval configurations
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 200
TOP_K = 5
BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5

# RAG Generation Prompt
DEFAULT_SYSTEM_PROMPT = """You are an expert AI assistant specializing in Machine Learning and Data Science.
Your task is to answer the user's question accurately, clearly, and concisely using ONLY the provided context documents.

Guidelines:
1. Base your answer strictly on the provided Context. Do NOT hallucinate, extrapolate, or invent information.
2. If the answer cannot be determined from the context, state: "I cannot find sufficient information in the provided documents to answer this question."
3. Cite the source document and page number when stating key facts (e.g., "[Source: ML - Regression.pdf, Page 26]").
4. Structure your response with bullet points and clear explanations where appropriate.

Context:
{context}

Question:
{question}
"""
