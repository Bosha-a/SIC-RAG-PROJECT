import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.utils import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
)

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    """
    Load all supported documents (.pdf, .txt) from the data directory.
    """
    documents = []

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file in sorted(data_dir.iterdir()):
        if file.suffix.lower() == ".pdf":
            loader = PyMuPDFLoader(str(file))
        elif file.suffix.lower() == ".txt":
            loader = TextLoader(
                str(file),
                encoding="utf-8"
            )
        else:
            continue

        loaded = loader.load()

        for doc in loaded:
            doc.metadata["filename"] = file.name

        documents.extend(loaded)

    if not documents:
        raise ValueError(f"No supported documents found in {data_dir}")

    return documents


def get_chunks(
    documents: Optional[List[Document]] = None,
    data_dir: Path = DATA_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Split loaded documents into chunks with unique chunk IDs.
    """
    if documents is None:
        documents = load_documents(data_dir=data_dir)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    # Remove empty chunks
    chunks = [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]

    # Give every chunk a unique identifier
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(i) 

    return chunks


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} document sections")

    all_chunks = get_chunks(docs)
    print(f"Created {len(all_chunks)} chunks")