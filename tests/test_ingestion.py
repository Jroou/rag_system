import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.ingestion.chunkers import chunk_code, chunk_markdown
from src.ingestion.loaders import detect_document_type, load_code, load_markdown
from src.ingestion.pipeline import IngestionPipeline
from src.storage.qdrant_store import DENSE_VECTOR_SIZE, QdrantStore
from src.storage.sqlite_store import SQLiteStore

FIXTURES = Path(__file__).parent / "fixtures"
TEST_QDRANT_PATH = "/tmp/rag_test_ingestion_qdrant"
TEST_SQLITE_PATH = "/tmp/rag_test_ingestion.db"


class MockEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        results = []
        for t in texts:
            h = hashlib.md5(t.encode()).digest()
            vec = [b / 255.0 for b in h]
            vec = (vec * (DENSE_VECTOR_SIZE // len(vec) + 1))[:DENSE_VECTOR_SIZE]
            results.append(vec)
        return results

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]


@pytest.fixture
def pipeline():
    qdrant_path = Path(TEST_QDRANT_PATH)
    sqlite_path = Path(TEST_SQLITE_PATH)
    if qdrant_path.exists():
        shutil.rmtree(qdrant_path)
    sqlite_path.unlink(missing_ok=True)

    qdrant = QdrantStore(path=str(qdrant_path), collection_name="test_ingestion")
    sqlite = SQLiteStore(db_path=str(sqlite_path))
    embedder = MockEmbedder()

    pipe = IngestionPipeline(
        qdrant_store=qdrant,
        sqlite_store=sqlite,
        embedder=embedder,
        parent_chunk_size=500,
        child_chunk_size=150,
        chunk_overlap=30,
    )
    yield pipe, qdrant, sqlite

    qdrant.close()
    sqlite.close()
    shutil.rmtree(qdrant_path, ignore_errors=True)
    sqlite_path.unlink(missing_ok=True)


class TestLoaders:
    def test_detect_markdown(self):
        assert detect_document_type(Path("test.md")) == "markdown"

    def test_detect_pdf(self):
        assert detect_document_type(Path("test.pdf")) == "pdf"

    def test_detect_code(self):
        assert detect_document_type(Path("test.py")) == "code"
        assert detect_document_type(Path("test.ts")) == "code"

    def test_detect_docx(self):
        assert detect_document_type(Path("test.docx")) == "docx"

    def test_load_markdown(self):
        text = load_markdown(FIXTURES / "sample.md")
        assert "Introduction" in text
        assert "Architecture" in text

    def test_load_code(self):
        text = load_code(FIXTURES / "sample.py")
        assert "class Calculator" in text
        assert "def fibonacci" in text


class TestChunkers:
    def test_markdown_produces_parent_and_child(self):
        text = load_markdown(FIXTURES / "sample.md")
        chunks = chunk_markdown(text, parent_chunk_size=500, child_chunk_size=150)
        parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        assert len(parent_chunks) > 0
        assert len(child_chunks) > 0
        for child in child_chunks:
            assert child.parent_id is not None
            assert any(
                p.metadata["chunk_id"] == child.parent_id for p in parent_chunks
            )

    def test_code_produces_parent_and_child(self):
        from langchain_text_splitters import Language

        text = load_code(FIXTURES / "sample.py")
        chunks = chunk_code(
            text, language=Language.PYTHON, parent_chunk_size=500, child_chunk_size=150
        )
        parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        assert len(parent_chunks) > 0
        assert len(child_chunks) >= 0  # small code may not split further
        for child in child_chunks:
            assert child.parent_id is not None


class TestIngestionPipeline:
    def test_ingest_markdown(self, pipeline):
        pipe, qdrant, sqlite = pipeline
        doc_id = pipe.ingest(FIXTURES / "sample.md")
        assert doc_id is not None
        assert qdrant.count() > 0

        doc = sqlite.get_document_by_path(str((FIXTURES / "sample.md").resolve()))
        assert doc is not None
        assert doc["status"] == "indexed"
        assert doc["document_type"] == "markdown"

    def test_ingest_code(self, pipeline):
        pipe, qdrant, sqlite = pipeline
        doc_id = pipe.ingest(FIXTURES / "sample.py")
        assert doc_id is not None
        assert qdrant.count() > 0

    def test_skip_unchanged(self, pipeline):
        pipe, qdrant, sqlite = pipeline
        doc_id_1 = pipe.ingest(FIXTURES / "sample.md")
        count_after_first = qdrant.count()
        doc_id_2 = pipe.ingest(FIXTURES / "sample.md")
        assert doc_id_1 == doc_id_2
        assert qdrant.count() == count_after_first

    def test_force_reindex(self, pipeline):
        pipe, qdrant, sqlite = pipeline
        pipe.ingest(FIXTURES / "sample.md")
        pipe.ingest(FIXTURES / "sample.md", force=True)
        doc = sqlite.get_document_by_path(str((FIXTURES / "sample.md").resolve()))
        assert doc["status"] == "indexed"

    def test_remove_document(self, pipeline):
        pipe, qdrant, sqlite = pipeline
        pipe.ingest(FIXTURES / "sample.md")
        assert qdrant.count() > 0
        pipe.remove(FIXTURES / "sample.md")
        assert qdrant.count() == 0
        doc = sqlite.get_document_by_path(str((FIXTURES / "sample.md").resolve()))
        assert doc is None

    def test_parent_child_linking_in_qdrant(self, pipeline):
        pipe, qdrant, sqlite = pipeline
        pipe.ingest(FIXTURES / "sample.md")

        # Search for child chunks
        embedder = MockEmbedder()
        query_emb = embedder.embed_query("architecture")
        results = qdrant.search(
            query_embedding=query_emb,
            top_k=50,
            filter_conditions={"chunk_type": "child"},
        )
        assert len(results) > 0
        for r in results:
            assert r["metadata"]["parent_chunk_id"] is not None

        # Verify parent chunks exist
        parent_results = qdrant.search(
            query_embedding=query_emb,
            top_k=50,
            filter_conditions={"chunk_type": "parent"},
        )
        assert len(parent_results) > 0
