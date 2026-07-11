import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.base import RetrievalResult
from src.retrieval.reranker import Reranker
from src.retrieval.semantic import SemanticStrategy
from src.generation.generator import Generator
from src.core.rag_engine import RAGEngine
from src.routing.router import Router
from src.storage.qdrant_store import QdrantStore

_TEST_VECTOR_SIZE = 1024
from src.storage.sqlite_store import SQLiteStore

FIXTURES = Path(__file__).parent / "fixtures"
TEST_QDRANT_PATH = "/tmp/rag_test_retrieval_qdrant"
TEST_SQLITE_PATH = "/tmp/rag_test_retrieval.db"


class MockEmbedder:
    """Deterministic embedder based on text hash for testing."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        results = []
        for t in texts:
            h = hashlib.md5(t.encode()).digest()
            vec = [b / 255.0 for b in h]
            vec = (vec * (_TEST_VECTOR_SIZE // len(vec) + 1))[:_TEST_VECTOR_SIZE]
            results.append(vec)
        return results

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]


@pytest.fixture
def setup():
    qdrant_path = Path(TEST_QDRANT_PATH)
    sqlite_path = Path(TEST_SQLITE_PATH)
    if qdrant_path.exists():
        shutil.rmtree(qdrant_path)
    sqlite_path.unlink(missing_ok=True)

    qdrant = QdrantStore(path=str(qdrant_path), collection_name="test_retrieval", vector_size=_TEST_VECTOR_SIZE)
    sqlite = SQLiteStore(db_path=str(sqlite_path))
    embedder = MockEmbedder()

    config = {
        "chunking": {
            "parent_chunk_size": 500,
            "child_chunk_size": 150,
            "chunk_overlap": 30,
        }
    }

    pipeline = IngestionPipeline(
        qdrant_store=qdrant, sqlite_store=sqlite, embedder=embedder, config=config
    )
    pipeline.ingest(FIXTURES / "sample.md")
    pipeline.ingest(FIXTURES / "sample.py")

    strategy = SemanticStrategy(qdrant_store=qdrant, embedder=embedder)

    yield strategy, qdrant, embedder

    qdrant.close()
    sqlite.close()
    shutil.rmtree(qdrant_path, ignore_errors=True)
    sqlite_path.unlink(missing_ok=True)


class TestSemanticStrategy:
    def test_retrieve_returns_results(self, setup):
        strategy, qdrant, embedder = setup
        results = strategy.retrieve("architecture", top_k=5)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_results_have_metadata(self, setup):
        strategy, qdrant, embedder = setup
        results = strategy.retrieve("modular architecture", top_k=5)
        for r in results:
            assert r.source_path != ""
            assert r.text != ""
            assert r.chunk_id != ""

    def test_results_are_scored(self, setup):
        strategy, qdrant, embedder = setup
        results = strategy.retrieve("fibonacci", top_k=5)
        assert all(r.score >= 0 for r in results)


class TestGenerator:
    def test_build_prompt_structure(self):
        mock_llm = MagicMock()
        generator = Generator(llm=mock_llm, system_prompt="You are a helper.")

        results = [
            RetrievalResult(
                chunk_id="c1",
                text="The sky is blue.",
                score=0.9,
                source_path="/docs/sky.md",
                document_type="markdown",
                parent_text="The sky is blue. It is also vast.",
            )
        ]

        messages = generator.build_prompt("What color is the sky?", results)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith("You are a helper.")
        assert messages[-1]["role"] == "user"
        assert "What color is the sky?" in messages[-1]["content"]
        assert "/docs/sky.md" in messages[-1]["content"]
        assert "The sky is blue. It is also vast." in messages[-1]["content"]

    def test_generate_calls_llm(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="The sky is blue.")
        generator = Generator(llm=mock_llm, system_prompt="Helper")

        results = [
            RetrievalResult(
                chunk_id="c1",
                text="Blue sky",
                score=0.9,
                source_path="/sky.md",
                document_type="markdown",
            )
        ]

        answer = generator.generate("What color?", results)
        assert answer == "The sky is blue."
        mock_llm.invoke.assert_called_once()


class MockReranker:
    def rerank(self, query, results, top_n=5):
        return results[:top_n]


class TestRAGEngine:
    def test_query_end_to_end(self, setup):
        strategy, qdrant, embedder = setup
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="The system uses modular architecture."
        )
        generator = Generator(llm=mock_llm, system_prompt="You are a helper.")
        router = Router(strategies={"semantic": strategy})
        reranker = MockReranker()
        engine = RAGEngine(router=router, reranker=reranker, generator=generator, top_k=5)

        answer, results, strategy_name = engine.query("architecture")
        assert "modular architecture" in answer
        assert len(results) > 0
        assert strategy_name == "semantic"
        mock_llm.invoke.assert_called_once()

    def test_chat_history_maintained(self, setup):
        strategy, qdrant, embedder = setup
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer 1")
        generator = Generator(llm=mock_llm, system_prompt="Helper")
        router = Router(strategies={"semantic": strategy})
        reranker = MockReranker()
        engine = RAGEngine(router=router, reranker=reranker, generator=generator, top_k=5)

        engine.query("First question")
        assert len(engine._chat_history) == 2

        mock_llm.invoke.return_value = MagicMock(content="Answer 2")
        engine.query("Second question")
        assert len(engine._chat_history) == 4
