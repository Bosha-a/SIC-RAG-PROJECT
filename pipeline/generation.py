import sys
from pathlib import Path

from typing import Dict, List, Optional
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from config.utils import (
    DEFAULT_SYSTEM_PROMPT,
    GEMINI_MODEL_NAME,
    GOOGLE_API_KEY,
    TOP_K,
)
from pipeline.retrieval import HybridRetrieverPipeline

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_llm(
    model_name: str = GEMINI_MODEL_NAME,
    temperature: float = 0.0,
    api_key: Optional[str] = GOOGLE_API_KEY
) -> ChatGoogleGenerativeAI:
    """
    Initialize Google Gemini Chat Model.
    """
    key = api_key or GOOGLE_API_KEY
    if not key:
        raise ValueError("GOOGLE_API_KEY is not set in environment or .env file.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=key,
        temperature=temperature
    )


def format_context(docs: List[Document]) -> str:
    """
    Format retrieved documents into a structured context string for the prompt.
    """
    context_blocks = []
    for doc in docs:
        filename = doc.metadata.get("filename", "Unknown document")
        page = doc.metadata.get("page", "N/A")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        block = f"--- [Document: {filename} | Page: {page} | Chunk ID: {chunk_id}] ---\n{doc.page_content.strip()}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def generate_answer(
    query: str,
    retriever: Optional[HybridRetrieverPipeline] = None,
    k: int = TOP_K,
    mode: str = "hybrid",
    model_name: str = GEMINI_MODEL_NAME,
    llm: Optional[ChatGoogleGenerativeAI] = None
) -> Dict:
    """
    Retrieve relevant documents and generate an answer using the LLM with the static system prompt.
    """
    if retriever is None:
        retriever = HybridRetrieverPipeline()

    docs = retriever.search(query=query, k=k, mode=mode)

    if not docs:
        return {
            "query": query,
            "answer": "No relevant documents found in the database.",
            "sources": []
        }

    formatted_context = format_context(docs)

    # Use the static system prompt
    final_prompt = DEFAULT_SYSTEM_PROMPT.format(
        context=formatted_context,
        question=query
    )

    if llm is None:
        llm = get_llm(model_name=model_name)

    response = llm.invoke(final_prompt)
    answer_text = response.content

    # If response content is a list of blocks, extract text
    if isinstance(answer_text, list):
        answer_text = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in answer_text)

    sources = [
        {
            "filename": doc.metadata.get("filename"),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "snippet": doc.page_content[:200]
        }
        for doc in docs
    ]

    return {
        "query": query,
        "answer": answer_text,
        "sources": sources
    }


if __name__ == "__main__":
    test_query = "What is the difference between simple and multiple linear regression?"
    print(f"Query: {test_query}\n")
    print("Generating answer using Gemini with System Prompt...\n")

    result = generate_answer(query=test_query, k=3)
    print("=" * 70)
    print("GENERATED ANSWER:")
    print("=" * 70)
    print(result["answer"])
    print("\n" + "=" * 70)
    print("SOURCES USED:")
    print("=" * 70)
    for src in result["sources"]:
        print(f"- {src['filename']} (Page {src['page']}, Chunk {src['chunk_id']})")
