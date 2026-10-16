import sys
from pathlib import Path

from typing import Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

from config.utils import TOP_K, BM25_WEIGHT, VECTOR_WEIGHT
from pipeline.chunking import get_chunks
from pipeline.embedding import get_or_create_vectorstore

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    

def get_keyword_retriever(
    chunks: Optional[List[Document]] = None,
    k: int = TOP_K
) -> BM25Retriever:
    """
    Build BM25 keyword retriever from document chunks.
    """
    if chunks is None:
        chunks = get_chunks()

    retriever = BM25Retriever.from_documents( 
        chunks,
        preprocess_func=lambda text: text.lower().split(),
        k=k
    )
    return retriever


def get_vector_retriever(
    vectorstore: Optional[Chroma] = None,
    k: int = TOP_K
):
    """
    Build vector similarity retriever from Chroma vectorstore.
    """
    if vectorstore is None:
        vectorstore = get_or_create_vectorstore()

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def hybrid_search(
    query: str,
    keyword_retriever: Optional[BM25Retriever] = None,
    vector_retriever = None,
    k: int = TOP_K,
    weight_keyword: float = BM25_WEIGHT,
    weight_vector: float = VECTOR_WEIGHT
) -> List[Document]:
    """
    Perform hybrid search combining BM25 keyword and dense vector retrieval using RRF.
    """
    if keyword_retriever is None:
        keyword_retriever = get_keyword_retriever(k=k)
    if vector_retriever is None:
        vector_retriever = get_vector_retriever(k=k)

    keyword_docs = keyword_retriever.invoke(query) # keyword 
    vector_docs = vector_retriever.invoke(query) # vector 

    scores: Dict[Tuple, float] = {}
    doc_map: Dict[Tuple, Document] = {}

    def add_results(docs: List[Document], weight: float):
        for rank, doc in enumerate(docs, start=1):
            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.metadata.get("chunk_id")
            )
            scores[key] = scores.get(key, 0.0) + (weight / (60 + rank))
            doc_map[key] = doc

    add_results(keyword_docs, weight=weight_keyword)
    add_results(vector_docs, weight=weight_vector)

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[key] for key in ranked_keys[:k]]


class HybridRetrieverPipeline:
    """
    High-level manager caching chunks, vectorstore, and retrievers for rapid querying.
    """
    def __init__(self, chunks: Optional[List[Document]] = None, vectorstore: Optional[Chroma] = None):
        if chunks is None:
            self.chunks = get_chunks()
        else:
            self.chunks = chunks

        if vectorstore is None:
            self.vectorstore = get_or_create_vectorstore(chunks=self.chunks)
        else:
            self.vectorstore = vectorstore

        self.keyword_retriever = get_keyword_retriever(chunks=self.chunks)
        self.vector_retriever = get_vector_retriever(vectorstore=self.vectorstore)

    def search(self, query: str, k: int = TOP_K, mode: str = "hybrid") -> List[Document]:
        if mode == "keyword":
            return self.keyword_retriever.invoke(query)[:k]
        elif mode == "vector":
            return self.vector_retriever.invoke(query)[:k]
        else:
            return hybrid_search(
                query=query,
                keyword_retriever=self.keyword_retriever,
                vector_retriever=self.vector_retriever,
                k=k
            )


if __name__ == "__main__":
    query = "What are the main requirements?"

    pipeline = HybridRetrieverPipeline()

    print("\n========== KEYWORD SEARCH ==========")
    keyword_results = pipeline.search(query, mode="keyword")
    for doc in keyword_results:
        print("Source:", doc.metadata.get("filename", "unknown"))
        print(doc.page_content[:300])
        print("-" * 50)

    print("\n========== VECTOR SEARCH ==========")
    vector_results = pipeline.search(query, mode="vector")
    for doc in vector_results:
        print("Source:", doc.metadata.get("filename", "unknown"))
        print(doc.page_content[:300])
        print("-" * 50)

    print("\n========== HYBRID SEARCH ==========")
    hybrid_results = pipeline.search(query, mode="hybrid")
    for doc in hybrid_results:
        print("Source:", doc.metadata.get("filename", "unknown"))
        print(doc.page_content[:300])
        print("-" * 50)