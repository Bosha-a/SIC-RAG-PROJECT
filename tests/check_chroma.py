
import sys
import json
import argparse
from pathlib import Path
from collections import Counter
from pipeline.embedding import get_or_create_vectorstore
from config.utils import CHROMA_DIR, COLLECTION_NAME

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")




def inspect_vector_db(
    limit: int = 5,
    filter_filename: str = None,
    filter_page: int = None,
    show_full: bool = False,
    summary_only: bool = False,
    as_json: bool = False,
):
    # Load vectorstore
    vectorstore = get_or_create_vectorstore()
    collection = vectorstore._collection

    total_count = collection.count()

    if not as_json:
        print("=" * 70)
        print(" CHROMA VECTOR DATABASE INSPECTION TOOL")
        print(f" Directory  : {CHROMA_DIR}")
        print(f" Collection : {COLLECTION_NAME}")
        print("=" * 70)
        print(f"Total Vectors in Database: {total_count}\n")

    if total_count == 0:
        if as_json:
            print("[]")
        else:
            print("Database is empty.")
        return

    # Fetch all records
    data = collection.get(
        include=["documents", "metadatas"]
    )

    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    # Group chunks by filename
    file_counts = Counter(
        (meta or {}).get("filename", "unknown")
        for meta in metas
    )

    if not as_json:
        print("--- Document Chunk Breakdown ---")

        for fname, count in sorted(file_counts.items()):
            print(f"{fname}: {count} chunks")

        print("-" * 70)

    if summary_only:
        if as_json:
            print(json.dumps({
                "total_vectors": total_count,
                "file_counts": dict(file_counts)
            }, indent=2, ensure_ascii=False))
        return

    # Build records and apply filters
    records = []

    for record_id, doc, meta in zip(ids, docs, metas):
        meta = meta or {}
        doc = doc or ""

        # Filter by filename
        if filter_filename is not None:
            filename = str(meta.get("filename", ""))

            if filter_filename.lower() not in filename.lower():
                continue

        # Filter by page number
        if filter_page is not None:
            if str(meta.get("page")) != str(filter_page):
                continue

        records.append({
            "id": record_id,
            "document": doc,
            "metadata": meta
        })

    if not as_json:
        print(f"\nMatching records: {len(records)}")

    # Apply display limit
    display_records = (
        records if limit <= 0 else records[:limit]
    )

    # JSON output
    if as_json:
        print(json.dumps(
            display_records,
            indent=2,
            ensure_ascii=False,
            default=str
        ))
        return

    # Display matching records
    for idx, item in enumerate(display_records, start=1):

        meta = item["metadata"]
        content = item["document"]

        print(f"\n[Record {idx}/{len(records)}]")
        print(f"ID       : {item['id']}")
        print(f"File     : {meta.get('filename', 'N/A')}")
        print(
            f"Page     : {meta.get('page', 'N/A')} "
            f"(of {meta.get('total_pages', 'N/A')})"
        )
        print(f"Chunk ID : {meta.get('chunk_id', 'N/A')}")

        print("Metadata :")

        for key, value in meta.items():
            if key not in [
                "filename",
                "page",
                "chunk_id",
                "total_pages"
            ]:
                print(f"  - {key}: {value}")

        print("Content  :")

        if show_full:
            print(content)
        else:
            text = content.strip()
            snippet = text[:300]

            if len(text) > 300:
                snippet += "..."

            print(f"  {snippet}")

        print("-" * 70)

    if limit > 0 and len(records) > limit:
        print(
            f"\nShowing {limit} of {len(records)} records. "
            "Use '--limit 0' to display all."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Inspect Chroma vector database."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of records (0 for all)"
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Filter by filename substring"
    )

    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="Filter by page number"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Display full chunk content"
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Display chunk counts only"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON"
    )

    args = parser.parse_args()

    inspect_vector_db(
        limit=args.limit,
        filter_filename=args.file,
        filter_page=args.page,
        show_full=args.full,
        summary_only=args.summary,
        as_json=args.json,
    )
