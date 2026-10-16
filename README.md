# 🤖 End-to-End RAG System & Intelligence Hub

An enterprise-ready **Retrieval-Augmented Generation (RAG)** system designed for document search and context-grounded question answering. Built with **LangChain**, **ChromaDB**, **Hybrid Search (BM25 + Dense Vectors via RRF)**, and powered by **Google Gemini** with strict citation-backed responses.

Provides three distinct interfaces: an interactive **Streamlit Web UI**, a high-performance **FastAPI REST API**, and a versatile **Command-Line Interface (CLI)**.

---

## 📑 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
- [Running the Application](#-running-the-application)
  - [1. Interactive Streamlit Web UI](#1-interactive-streamlit-web-ui)
  - [2. FastAPI Server & REST API](#2-fastapi-server--rest-api)
  - [3. Command-Line Interface (CLI)](#3-command-line-interface-cli)
  - [4. Inspecting the Chroma Vector Database](#4-inspecting-the-chroma-vector-database)
- [API Reference](#-api-reference)
- [Configuration Settings](#-configuration-settings)
- [Adding New Documents](#-adding-new-documents)
- [License](#-license)

---

## 🏗 Overview & Architecture

The system indexes unstructured documents (PDFs, text files), splits them into overlapping semantic chunks, embeds them using **Jina AI Embeddings** (or local HuggingFace fallbacks), and stores them into **ChromaDB**. 

When a query is received:
1. **Hybrid Retrieval**: Queries both a **BM25 Keyword Retriever** (lexical exact match) and a **Dense Vector Retriever** (semantic similarity).
2. **Reciprocal Rank Fusion (RRF)**: Combines and ranks documents based on rank positions to provide optimal recall and precision.
3. **Grounded Generation**: Feeds top context chunks to **Google Gemini** with a strict system prompt preventing hallucinations and mandating source citations.

```
                              [ Documents (PDF / TXT) ]
                                          │
                                          ▼
                            [ Document Ingestion & Chunking ]
                           (PyMuPDF / Recursive Splitter)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         [ ChromaDB Vectorstore ]                         [ BM25 Retriever ]
         (Jina AI Embeddings)                             (Sparse Lexical)
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │ User Query
                                          ▼
                             [ Reciprocal Rank Fusion (RRF) ]
                                  (Hybrid Search Engine)
                                          │
                                          ▼
                             [ Context + Prompt Injection ]
                                          │
                                          ▼
                             [ Google Gemini (LLM) ]
                                          │
                                          ▼
                        [ Grounded Answer + Source Citations ]
                                 (UI / REST API / CLI)
```

---

## ✨ Key Features

- **Hybrid Search with RRF**: Balances exact term matching (BM25) with deep semantic context (Dense Vectors) using weighted Reciprocal Rank Fusion.
- **Strict Anti-Hallucination Guardrails**: System prompts enforce answering *strictly* based on context, with explicit fallback handling when information is missing.
- **Traceable Citations**: Every generated answer cites the specific document filename, page number, and chunk ID.
- **Multiple Interfaces**:
  - **Streamlit Web UI**: Visual question answering, document search, and real-time ChromaDB vector inspection.
  - **FastAPI Backend**: Complete REST endpoints with automatic OpenAPI / Swagger interactive documentation (`/docs`).
  - **CLI Mode**: Query and test directly from the terminal without launching servers.
- **Flexible Embeddings**: Plug-and-play support for Jina AI API embeddings (`jina-embeddings-v2-base-en`) with automatic fallback to local HuggingFace `sentence-transformers`.
- **Database Inspection Tools**: Built-in CLI scripts to inspect metadata, page distribution, and chunk health.

---

## 🛠 Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Orchestration** | [LangChain](https://www.langchain.com/) | Document loading, chunking, retrieval orchestration, and LLM chains |
| **LLM Generator** | [Google Gemini](https://ai.google.dev/) (`gemini-3.8-flash`) | Context synthesis via `langchain-google-genai` |
| **Embeddings** | [Jina AI](https://jina.ai/) / [HuggingFace](https://huggingface.co/) | `jina-embeddings-v2-base-en` or `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | Persistent vector database stored on disk |
| **Lexical Search** | [Rank-BM25](https://github.com/dorianbrown/rank_bm25) | BM25 sparse keyword indexing |
| **Web UI** | [Streamlit](https://streamlit.io/) | Interactive dashboard with tabs for Q&A, Search, and Vector Explorer |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) & Uvicorn | Production REST API with CORS and Pydantic validation |
| **Document Loaders** | PyMuPDF (`fitz`), Python-Docx | High-speed PDF and text extraction |

---

## 📁 Project Directory Structure

```text
End_to_End_Project/
├── config/
│   └── utils.py               # Paths, model parameters, constants, and system prompts
├── data/                      # Source documents (.pdf, .txt)
│   ├── Decision Tree & Ensemble Methods.pdf
│   ├── Logistic Regression & KNN.pdf
│   ├── ML - Regression.pdf
│   └── ML - Unsuprevised learning - Clustering.pdf
├── models/
│   ├── __init__.py
│   └── schemas.py             # Pydantic schemas for API requests & responses
├── notebooks/                 # Jupyter notebooks for experiments and prototyping
├── pipeline/
│   ├── __init__.py
│   ├── chunking.py            # Document loading and RecursiveCharacterTextSplitter
│   ├── embedding.py           # Embedding initialization & ChromaDB vectorstore management
│   ├── generation.py          # Gemini LLM answer generation with citation formatting
│   └── retrieval.py           # BM25, Dense Vector, and RRF Hybrid search implementations
├── routes/
│   ├── __init__.py
│   └── endpoints.py           # FastAPI routes (/api/status, /api/search, /api/ask, /api/reindex)
├── tests/
│   ├── check_chroma.py        # CLI diagnostic tool to inspect ChromaDB collections
│   └── check_embedding_model.py # Embedding connection test
├── chroma_db/                 # Persistent on-disk vector store (auto-generated)
├── main.py                    # Unified entry point (FastAPI, CLI, and Streamlit launcher)
├── streamlit_app.py           # Streamlit Web Application
├── requirements.txt           # Project dependencies
├── .env.example               # Example environment variable file
├── .env                       # Local environment variables (API keys)
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: Version `3.10` or higher (`3.11` / `3.12` recommended)
- **API Keys**:
  - [Google Gemini API Key](https://aistudio.google.com/) (Required for answer generation)
  - [Jina AI API Key](https://jina.ai/embeddings/) (Optional; defaults to HuggingFace local models if not provided)

### Installation

1. **Clone the repository** (or navigate to project directory):
   ```bash
   cd End_to_End_Project
   ```

2. **Create and activate a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Environment Configuration

Create a `.env` file in the root directory by copying `.env.example`:

```bash
# On Windows PowerShell
cp .env.example .env
```

Edit `.env` and supply your API keys:

```ini
GOOGLE_API_KEY="your_google_ai_studio_api_key_here"
JINA_API_KEY="your_jina_api_key_here"

# Optional overrides:
# GEMINI_MODEL_NAME=gemini-3.8-flash
# JINA_MODEL_NAME=jina-embeddings-v2-base-en
# EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

---

## 💻 Running the Application

`main.py` serves as the unified execution hub.

### 1. Interactive Streamlit Web UI

Run the Streamlit application using the launcher:

```bash
python main.py --ui
```

*Or directly with Streamlit:*
```bash
streamlit run streamlit_app.py
```

Open your browser at **`http://localhost:8501`**.

The UI includes 3 main tabs:
- 💬 **Ask AI**: Enter any question; Gemini synthesizes an answer referencing the uploaded materials with collapsible source cards.
- 🔍 **Document Search**: Search directly through BM25, Vector, or Hybrid mode without calling the LLM.
- 📊 **Vector Database Explorer**: View total vectors, document distribution, and explore specific chunk contents and metadata.

---

### 2. FastAPI Server & REST API

Start the FastAPI application with auto-reload:

```bash
python main.py --port 8000 --host 127.0.0.1
```

*Or using Uvicorn directly:*
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive API Documentation (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative Docs (ReDoc)**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### 3. Command-Line Interface (CLI)

You can run fast CLI queries without starting a web server:

- **Ask a question with LLM Generation & Citations**:
  ```bash
  python main.py --ask "What is the difference between simple and multiple linear regression?"
  ```

- **Perform a retrieval search (hybrid document matching)**:
  ```bash
  python main.py --query "What are the assumptions of linear regression?"
  ```

- **Run a default smoke test**:
  ```bash
  python main.py --test
  ```

---

### 4. Inspecting the Chroma Vector Database

Inspect the stored embeddings, metadata, and chunk breakdown using `tests/check_chroma.py`:

```bash
# View summary breakdown of stored documents
python tests/check_chroma.py --summary

# Inspect top 5 chunks with metadata
python tests/check_chroma.py --limit 5

# Filter chunks by document name
python tests/check_chroma.py --file "Regression" --limit 3

# Display full content in JSON format
python tests/check_chroma.py --limit 2 --json
```

---

## 📡 API Reference

When the FastAPI server is running, the following endpoints are available:

### `GET /api/status`
Check pipeline readiness, chunk count, and ChromaDB persistence status.

**Response Example:**
```json
{
  "status": "ready",
  "total_chunks": 420,
  "chroma_persisted": true
}
```

---

### `POST /api/search`
Perform document search across indexed chunks.

**Request Body:**
```json
{
  "query": "K-Means centroid initialization",
  "k": 3,
  "mode": "hybrid"
}
```
> `mode` options: `"hybrid"` *(recommended)*, `"vector"`, or `"keyword"`.

**Response Example:**
```json
{
  "query": "K-Means centroid initialization",
  "mode": "hybrid",
  "total_results": 3,
  "results": [
    {
      "content": "...",
      "filename": "ML - Unsuprevised learning - Clustering.pdf",
      "page": 14,
      "chunk_id": "128"
    }
  ]
}
```

---

### `POST /api/ask`
Retrieve context chunks and generate a grounded answer using Google Gemini.

**Request Body:**
```json
{
  "query": "What are the advantages of DBSCAN over K-Means?",
  "k": 3,
  "mode": "hybrid"
}
```

**Response Example:**
```json
{
  "query": "What are the advantages of DBSCAN over K-Means?",
  "answer": "Based on the provided documents, DBSCAN offers the following advantages...\n[Source: ML - Unsuprevised learning - Clustering.pdf, Page 32]",
  "sources": [
    {
      "filename": "ML - Unsuprevised learning - Clustering.pdf",
      "page": 32,
      "chunk_id": "184",
      "snippet": "DBSCAN does not require specifying the number of clusters in advance..."
    }
  ]
}
```

---

### `POST /api/reindex`
Forces a reload of all files from `data/`, re-chunks them, and updates the ChromaDB vector store.

**Response Example:**
```json
{
  "status": "reindexed successfully",
  "total_chunks": 420
}
```

---

## ⚙️ Configuration Settings

Key configuration variables are defined in [config/utils.py](config/utils.py) and can be adjusted directly or via environment variables:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `CHUNK_SIZE` | `1000` | Target character size per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks to maintain context |
| `TOP_K` | `5` | Default number of chunks retrieved per query |
| `BM25_WEIGHT` | `0.5` | Weight factor for BM25 keyword rankings in RRF |
| `VECTOR_WEIGHT` | `0.5` | Weight factor for dense vector rankings in RRF |
| `GEMINI_MODEL_NAME` | `"gemini-3.8-flash"` | Google Gemini model for response generation |
| `JINA_MODEL_NAME` | `"jina-embeddings-v2-base-en"` | Jina AI embedding model |
| `COLLECTION_NAME` | `"rag_documents"` | ChromaDB collection identifier |

---

## 📂 Adding New Documents

1. Place your `.pdf` or `.txt` files into the `data/` directory.
2. Re-index the database using any of the following methods:
   - **Streamlit UI**: Click the **"🔄 Force Re-Index Documents"** button in the sidebar.
   - **FastAPI**: Trigger `POST http://127.0.0.1:8000/api/reindex`.
   - **Python CLI / Script**:
     ```bash
     python -c "from pipeline.chunking import get_chunks; from pipeline.embedding import get_or_create_vectorstore; get_or_create_vectorstore(chunks=get_chunks(), force_reindex=True)"
     ```

---

## 📄 License

This project is developed for educational and professional demonstration purposes. All original code is distributed under the MIT License.
