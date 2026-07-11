import uuid
from pathlib import Path

from src.ingestion.chunkers import (
    EXTENSION_TO_LANGUAGE,
    Chunk,
    chunk_code,
    chunk_docx,
    chunk_markdown,
    chunk_pdf,
)
from src.ingestion.loaders import (
    detect_document_type,
    extract_pdf_title,
    load_code,
    load_docx,
    load_markdown,
    load_pdf,
)
from src.storage.qdrant_store import QdrantStore
from src.storage.sqlite_store import SQLiteStore, compute_file_hash


class IngestionPipeline:
    def __init__(
        self,
        qdrant_store: QdrantStore,
        sqlite_store: SQLiteStore,
        embedder,
        *,
        parent_chunk_size: int = 1000,
        child_chunk_size: int = 200,
        chunk_overlap: int = 50,
    ):
        self._qdrant = qdrant_store
        self._sqlite = sqlite_store
        self._embedder = embedder
        self._parent_chunk_size = parent_chunk_size
        self._child_chunk_size = child_chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest(self, file_path: Path, force: bool = False, original_name: str | None = None, thread_id: str | None = None) -> str | None:
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            return None

        file_hash = compute_file_hash(file_path)
        existing = self._sqlite.get_document_by_path(str(file_path))

        if existing and existing["file_hash"] == file_hash and not force:
            return existing["id"]

        document_id = existing["id"] if existing else str(uuid.uuid4())

        if existing:
            self._qdrant.delete_by_document_id(document_id)

        doc_type = detect_document_type(file_path)
        if original_name is None and doc_type == "pdf":
            original_name = extract_pdf_title(file_path) or file_path.name
        try:
            chunks = self._chunk_document(file_path, doc_type, document_id)
        except Exception as e:
            self._sqlite.upsert_document(
                document_id=document_id,
                source_path=str(file_path),
                document_type=doc_type,
                file_hash=file_hash,
                status="error",
                error_message=str(e),
                original_name=original_name,
                thread_id=thread_id,
            )
            return None

        if not chunks:
            self._sqlite.upsert_document(
                document_id=document_id,
                source_path=str(file_path),
                document_type=doc_type,
                file_hash=file_hash,
                status="indexed",
                original_name=original_name,
                thread_id=thread_id,
            )
            return document_id

        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        parent_chunks = [c for c in chunks if c.chunk_type == "parent"]

        # Embed and store child chunks (for search)
        if child_chunks:
            child_texts = [c.text for c in child_chunks]
            child_embeddings = self._embedder.embed(child_texts)
            child_ids = [c.metadata["chunk_id"] for c in child_chunks]
            child_metadatas = [
                {
                    "document_id": document_id,
                    "parent_chunk_id": c.parent_id,
                    "chunk_type": "child",
                    "source_path": str(file_path),
                    "document_type": doc_type,
                    "text": c.text,
                    "thread_id": thread_id,
                }
                for c in child_chunks
            ]
            self._qdrant.add(
                ids=child_ids, embeddings=child_embeddings, metadatas=child_metadatas
            )

        # Store parent chunks (for generation context)
        if parent_chunks:
            parent_embeddings = self._embedder.embed([c.text for c in parent_chunks])
            parent_ids = [c.metadata["chunk_id"] for c in parent_chunks]
            parent_metadatas = [
                {
                    "document_id": document_id,
                    "parent_chunk_id": None,
                    "chunk_type": "parent",
                    "source_path": str(file_path),
                    "document_type": doc_type,
                    "text": c.text,
                    "thread_id": thread_id,
                }
                for c in parent_chunks
            ]
            self._qdrant.add(
                ids=parent_ids, embeddings=parent_embeddings, metadatas=parent_metadatas
            )

        self._sqlite.upsert_document(
            document_id=document_id,
            source_path=str(file_path),
            document_type=doc_type,
            file_hash=file_hash,
            status="indexed",
            original_name=original_name,
            thread_id=thread_id,
        )
        return document_id

    def remove(self, file_path: Path) -> None:
        doc_id = self._sqlite.delete_document(str(Path(file_path).resolve()))
        if doc_id:
            self._qdrant.delete_by_document_id(doc_id)

    def delete_thread_documents(self, thread_id: str) -> int:
        """Delete all Qdrant vectors and SQLite rows for documents scoped to a thread."""
        docs = self._sqlite.get_thread_documents(thread_id)
        for doc in docs:
            self._qdrant.delete_by_document_id(doc["id"])
            self._sqlite.delete_document(doc["source_path"])
        return len(docs)

    def promote_to_global(self, document_id: str) -> None:
        """Re-tag a thread-scoped document as global KB (thread_id = null) in both stores."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, PayloadUpdateOperation, SetPayloadOperation
        self._qdrant._client.set_payload(
            collection_name=self._qdrant._collection_name,
            payload={"thread_id": None},
            points=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]),
        )
        self._sqlite.promote_document_to_global(document_id)

    def _chunk_document(
        self, file_path: Path, doc_type: str, document_id: str
    ) -> list[Chunk]:
        base_metadata = {"document_id": document_id, "source_path": str(file_path)}

        if doc_type == "pdf":
            pages = load_pdf(file_path)
            return chunk_pdf(
                pages,
                self._parent_chunk_size,
                self._child_chunk_size,
                self._chunk_overlap,
                base_metadata,
            )
        elif doc_type == "markdown":
            text = load_markdown(file_path)
            return chunk_markdown(
                text,
                self._parent_chunk_size,
                self._child_chunk_size,
                self._chunk_overlap,
                base_metadata,
            )
        elif doc_type == "code":
            text = load_code(file_path)
            language = EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower())
            if language is None:
                return chunk_markdown(
                    text,
                    self._parent_chunk_size,
                    self._child_chunk_size,
                    self._chunk_overlap,
                    base_metadata,
                )
            return chunk_code(
                text,
                language,
                self._parent_chunk_size,
                self._child_chunk_size,
                self._chunk_overlap,
                base_metadata,
            )
        elif doc_type == "docx":
            paragraphs = load_docx(file_path)
            return chunk_docx(
                paragraphs,
                self._parent_chunk_size,
                self._child_chunk_size,
                self._chunk_overlap,
                base_metadata,
            )
        else:
            return []
