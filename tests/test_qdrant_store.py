import shutil
import uuid
from pathlib import Path

import pytest

from src.storage.qdrant_store import QdrantStore

_TEST_VECTOR_SIZE = 1024

TEST_QDRANT_PATH = "/tmp/rag_system_test_qdrant"
TEST_COLLECTION = "test_collection"


@pytest.fixture
def store():
    path = Path(TEST_QDRANT_PATH)
    if path.exists():
        shutil.rmtree(path)
    s = QdrantStore(path=TEST_QDRANT_PATH, collection_name=TEST_COLLECTION, vector_size=_TEST_VECTOR_SIZE)
    yield s
    s.close()
    shutil.rmtree(path, ignore_errors=True)


def _random_embedding() -> list[float]:
    import random

    random.seed(42)
    return [random.random() for _ in range(_TEST_VECTOR_SIZE)]


def _make_embedding(seed: int) -> list[float]:
    import random

    rng = random.Random(seed)
    return [rng.random() for _ in range(_TEST_VECTOR_SIZE)]


def test_add_and_search(store: QdrantStore):
    doc_id = "doc-1"
    chunk_id = str(uuid.uuid4())
    embedding = _make_embedding(1)

    store.add(
        ids=[chunk_id],
        embeddings=[embedding],
        metadatas=[
            {
                "document_id": doc_id,
                "parent_chunk_id": "parent-1",
                "chunk_type": "child",
                "source_path": "/docs/test.md",
                "document_type": "markdown",
                "language": "en",
            }
        ],
    )

    results = store.search(query_embedding=embedding, top_k=5)
    assert len(results) == 1
    assert results[0]["id"] == chunk_id
    assert results[0]["metadata"]["document_id"] == doc_id
    assert results[0]["metadata"]["chunk_type"] == "child"


def test_metadata_filtering(store: QdrantStore):
    emb1 = _make_embedding(10)
    emb2 = _make_embedding(20)

    store.add(
        ids=[str(uuid.uuid4()), str(uuid.uuid4())],
        embeddings=[emb1, emb2],
        metadatas=[
            {
                "document_id": "doc-1",
                "document_type": "markdown",
                "source_path": "/a.md",
            },
            {
                "document_id": "doc-2",
                "document_type": "pdf",
                "source_path": "/b.pdf",
            },
        ],
    )

    results = store.search(
        query_embedding=emb1,
        top_k=10,
        filter_conditions={"document_type": "pdf"},
    )
    assert len(results) == 1
    assert results[0]["metadata"]["document_type"] == "pdf"


def test_delete_by_document_id(store: QdrantStore):
    emb = _make_embedding(30)
    chunk_ids = [str(uuid.uuid4()) for _ in range(3)]

    store.add(
        ids=chunk_ids,
        embeddings=[emb, _make_embedding(31), _make_embedding(32)],
        metadatas=[
            {"document_id": "doc-to-delete", "source_path": f"/f{i}.md"}
            for i in range(3)
        ],
    )

    assert store.count() == 3
    store.delete_by_document_id("doc-to-delete")
    assert store.count() == 0


def test_fetch_parent_found(store: QdrantStore):
    parent_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())
    parent_emb = _make_embedding(60)
    child_emb = _make_embedding(61)

    store.add(
        ids=[parent_id, child_id],
        embeddings=[parent_emb, child_emb],
        metadatas=[
            {
                "document_id": "doc-fetch",
                "chunk_type": "parent",
                "text": "This is the parent text.",
                "source_path": "/doc.md",
            },
            {
                "document_id": "doc-fetch",
                "chunk_type": "child",
                "parent_chunk_id": parent_id,
                "text": "Child text.",
                "source_path": "/doc.md",
            },
        ],
    )

    result = store.fetch_parent(parent_id)
    assert result == "This is the parent text."


def test_fetch_parent_not_found(store: QdrantStore):
    # No data — should return None without raising
    result = store.fetch_parent("nonexistent-id")
    assert result is None


def test_fetch_parent_wrong_chunk_type(store: QdrantStore):
    # Only child chunk exists, fetch_parent should not return it
    child_id = str(uuid.uuid4())
    store.add(
        ids=[child_id],
        embeddings=[_make_embedding(62)],
        metadatas=[
            {
                "document_id": "doc-x",
                "chunk_type": "child",
                "text": "Child only.",
                "source_path": "/x.md",
            }
        ],
    )
    result = store.fetch_parent(child_id)
    assert result is None


def test_search_rrf_returns_only_children(store: QdrantStore):
    parent_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())
    query_emb = _make_embedding(70)

    store.add(
        ids=[parent_id, child_id],
        embeddings=[_make_embedding(71), query_emb],
        metadatas=[
            {
                "document_id": "doc-rrf",
                "chunk_type": "parent",
                "text": "Parent text.",
                "source_path": "/doc.md",
            },
            {
                "document_id": "doc-rrf",
                "chunk_type": "child",
                "parent_chunk_id": parent_id,
                "text": "Child text.",
                "source_path": "/doc.md",
            },
        ],
    )

    results = store.search_rrf(query_embedding=query_emb, top_k=10)
    assert len(results) == 1
    assert results[0]["metadata"]["chunk_type"] == "child"
    assert "id" in results[0]
    assert "score" in results[0]
    assert "metadata" in results[0]


def test_search_rrf_result_shape(store: QdrantStore):
    child_id = str(uuid.uuid4())
    emb = _make_embedding(80)

    store.add(
        ids=[child_id],
        embeddings=[emb],
        metadatas=[
            {
                "document_id": "doc-shape",
                "chunk_type": "child",
                "text": "Some child text.",
                "source_path": "/shape.md",
            }
        ],
    )

    results = store.search_rrf(query_embedding=emb, top_k=5)
    assert len(results) >= 1
    hit = results[0]
    assert set(hit.keys()) == {"id", "score", "metadata"}
    assert isinstance(hit["score"], float)
    assert isinstance(hit["metadata"], dict)


def test_search_rrf_empty_store(store: QdrantStore):
    results = store.search_rrf(query_embedding=_make_embedding(99), top_k=5)
    assert results == []


def test_persistence(tmp_path: Path):
    path = str(tmp_path / "qdrant_persist")
    s = QdrantStore(path=path, collection_name="persist_test", vector_size=_TEST_VECTOR_SIZE)
    emb = _make_embedding(50)
    chunk_id = str(uuid.uuid4())
    s.add(
        ids=[chunk_id],
        embeddings=[emb],
        metadatas=[{"document_id": "persist-doc", "source_path": "/x.md"}],
    )
    s.close()

    s2 = QdrantStore(path=path, collection_name="persist_test", vector_size=_TEST_VECTOR_SIZE)
    assert s2.count() == 1
    results = s2.search(query_embedding=emb, top_k=5)
    assert results[0]["metadata"]["document_id"] == "persist-doc"
    s2.close()
