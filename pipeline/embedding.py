import sys
from pathlib import Path

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config.utils import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    JINA_API_KEY,
    JINA_MODEL_NAME,
)
from pipeline.chunking import get_chunks

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))



def get_embedding_model(
    model_name: Optional[str] = None,
    jina_api_key: Optional[str] = JINA_API_KEY
) -> Embeddings:
    """
    Initialize and return the embedding model.
    Accesses Jina AI API if JINA_API_KEY is available;
    otherwise falls back to local HuggingFaceEmbeddings.
    """
    api_key = jina_api_key or JINA_API_KEY
    if api_key:
        target_model = model_name or JINA_MODEL_NAME or "jina-embeddings-v2-base-en"
        return JinaEmbeddings(
            jina_api_key=api_key,
            model_name=target_model
        )

    target_model = model_name or EMBEDDING_MODEL_NAME or "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(
        model_name=target_model,
        encode_kwargs={"normalize_embeddings": True}
    )


def get_or_create_vectorstore(
    chunks: Optional[List[Document]] = None,
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: Optional[Embeddings] = None,
    force_reindex: bool = False
) -> Chroma:
    """
    Retrieve existing Chroma vectorstore or create a new one from chunks.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    persist_dir_str = str(persist_directory)

    # If already indexed and not force_reindex, try loading existing ChromaDB
    if not force_reindex and persist_directory.exists():
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_model,
                persist_directory=persist_dir_str
            )
            if vectorstore._collection.count() > 0:
                return vectorstore
        except Exception:
            pass

    # Otherwise create vectorstore from chunks
    if chunks is None:
        chunks = get_chunks()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_dir_str
    )

    return vectorstore


if __name__ == "__main__":
    print("Initializing embeddings and vectorstore using Jina AI API...")
    chunks = get_chunks()
    vectorstore = get_or_create_vectorstore(chunks=chunks, force_reindex=True)
    print(f"Embeddings stored successfully in ChromaDB at {CHROMA_DIR}")