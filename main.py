import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.endpoints import router as api_router, get_pipeline

app = FastAPI(
    title="End-to-End RAG System",
    description="Hybrid Search and Document Retrieval Pipeline",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount pipeline endpoints
app.include_router(api_router)


@app.get("/")
def root():
    return {
        "message": "Welcome to the End-to-End RAG System API",
        "documentation": "/docs",
        "endpoints": {
            "status": "/api/status",
            "search": "/api/search",
            "ask": "/api/ask",
            "reindex": "/api/reindex"
        }
    }


def run_cli_test(query: str = "What are the main concepts in decision trees?"):
    print(f"\n--- Initializing Pipeline for query: '{query}' ---")
    pipeline = get_pipeline()
    results = pipeline.search(query=query, k=3, mode="hybrid")
    print(f"\nFound {len(results)} results:\n")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] File: {doc.metadata.get('filename')}, Page: {doc.metadata.get('page')}")
        print(f"Content:\n{doc.page_content[:250]}...\n")
        print("-" * 60)


def run_cli_ask(query: str):
    from pipeline.generation import generate_answer
    print(f"\nQuestion: {query}")
    print("Generating answer using Gemini with System Prompt...\n")
    pipeline = get_pipeline()
    res = generate_answer(query=query, retriever=pipeline, k=3)
    print("=" * 70)
    print("ANSWER:")
    print("=" * 70)
    print(res["answer"])
    print("\n" + "=" * 70)
    print("SOURCES:")
    print("=" * 70)
    for s in res["sources"]:
        print(f"- {s['filename']} (Page {s['page']}, Chunk {s['chunk_id']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End RAG Application")
    parser.add_argument("--test", action="store_true", help="Run a test query in CLI mode")
    parser.add_argument("--query", type=str, default=None, help="Run a retrieval search in CLI mode")
    parser.add_argument("--ask", type=str, default=None, help="Ask a question and generate an answer using LLM & System Prompt")
    parser.add_argument("--ui", action="store_true", help="Launch the Streamlit interactive Web UI")
    parser.add_argument("--port", type=int, default=8000, help="Port to run FastAPI server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind FastAPI server to")

    args = parser.parse_args()

    if args.ui:
        import subprocess
        print("Launching Streamlit UI at http://localhost:8501...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    elif args.ask:
        run_cli_ask(args.ask)
    elif args.query:
        run_cli_test(args.query)
    elif args.test:
        run_cli_test()
    else:
        print(f"Starting FastAPI server on http://{args.host}:{args.port} (Docs at http://{args.host}:{args.port}/docs)...")
        uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
