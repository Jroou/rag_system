"""Index the golden dataset into a dedicated eval Qdrant + SQLite store."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import IngestionPipeline
from src.storage.qdrant_store import QdrantStore
from src.storage.sqlite_store import SQLiteStore

GOLDEN_DS_PATH = Path.home() / "Documents" / "rag-golden-ds"
EVAL_DIR = Path(__file__).resolve().parent
QDRANT_PATH = str(EVAL_DIR / "qdrant_eval_data")
SQLITE_PATH = str(EVAL_DIR / "eval.db")
COLLECTION_NAME = "eval_knowledge_base"


def main():
    print("Initializing stores...", flush=True)
    qdrant = QdrantStore(path=QDRANT_PATH, collection_name=COLLECTION_NAME)
    sqlite = SQLiteStore(db_path=SQLITE_PATH)
    embedder = Embedder(device="cpu")

    pipeline = IngestionPipeline(
        qdrant_store=qdrant,
        sqlite_store=sqlite,
        embedder=embedder,
    )

    pdf_files = list(GOLDEN_DS_PATH.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to index", flush=True)

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"  [{i}/{len(pdf_files)}] Indexing: {pdf_path.name}", flush=True)
        doc_id = pipeline.ingest(pdf_path, force=True)
        if doc_id:
            print(f"    -> OK (id={doc_id[:8]}...)", flush=True)
        else:
            print(f"    -> FAILED", flush=True)

    print(f"\nDone. {qdrant.count()} chunks in Qdrant, "
          f"{len(sqlite.list_documents())} documents in SQLite.", flush=True)
    qdrant.close()
    sqlite.close()


if __name__ == "__main__":
    main()
