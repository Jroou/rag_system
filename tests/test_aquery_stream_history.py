import asyncio
from unittest.mock import MagicMock

from src.core.rag_engine import RAGEngine
from src.retrieval.base import RetrievalResult


class FakeStreamGenerator:
    """Generator whose agenerate_stream yields a fixed sequence of tokens."""

    def __init__(self, tokens: list[str]):
        self._tokens = tokens

    async def agenerate_stream(self, query, results, chat_history=None):
        for token in self._tokens:
            yield token


def test_aquery_stream_appends_both_turns_to_history():
    from src.retrieval.reranker import Reranker
    from src.routing.router import Router

    mock_result = RetrievalResult(
        chunk_id="c1",
        text="Some content",
        score=0.9,
        source_path="/docs/test.md",
        document_type="markdown",
        parent_text="Full parent content",
    )

    mock_strategy = MagicMock()
    mock_strategy.retrieve.return_value = [mock_result]

    router = MagicMock(spec=Router)
    router.route.return_value = ("semantic", mock_strategy)

    reranker = MagicMock(spec=Reranker)
    reranker.rerank.return_value = [mock_result]

    generator = FakeStreamGenerator(["Hello", ", ", "world", "!"])

    engine = RAGEngine(
        router=router,
        reranker=reranker,
        generator=generator,
        top_k=20,
        rerank_top_n=5,
    )

    async def run():
        stream, results, strategy = await engine.aquery_stream("What is this?")
        tokens = []
        async for token in stream:
            tokens.append(token)
        return "".join(tokens)

    full_response = asyncio.run(run())

    assert full_response == "Hello, world!"

    history = engine._chat_history
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "What is this?"}
    assert history[1] == {"role": "assistant", "content": "Hello, world!"}
