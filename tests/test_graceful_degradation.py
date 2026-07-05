import asyncio
from unittest.mock import MagicMock, patch

from src.core.rag_engine import RAGEngine
from src.retrieval.base import RetrievalResult


class FakeLLMError:
    async def astream(self, messages):
        raise ConnectionError("LLM API unavailable")
        yield  # make this an async generator


def test_fallback_stream_on_llm_failure():
    from src.generation.generator import Generator
    from src.retrieval.reranker import Reranker
    from src.routing.router import Router

    mock_strategy = MagicMock()
    mock_strategy.retrieve.return_value = [
        RetrievalResult(
            chunk_id="c1",
            text="Test chunk content",
            score=0.9,
            source_path="/docs/test.md",
            document_type="markdown",
            parent_text="Full parent content for test",
        )
    ]

    router = MagicMock(spec=Router)
    router.route.return_value = ("semantic", mock_strategy)

    reranker = MagicMock(spec=Reranker)
    reranker.rerank.return_value = mock_strategy.retrieve.return_value

    llm = FakeLLMError()
    generator = Generator(llm=llm, system_prompt="test")

    engine = RAGEngine(
        router=router,
        reranker=reranker,
        generator=generator,
        top_k=20,
        rerank_top_n=5,
    )

    async def run():
        stream, results, strategy = await engine.aquery_stream("test question")
        tokens = []
        async for t in stream:
            tokens.append(t)
        return "".join(tokens), results, strategy

    response, results, strategy = asyncio.run(run())
    assert "LLM unavailable" in response
    assert "raw results" in response
    assert "/docs/test.md" in response
    assert len(results) == 1
