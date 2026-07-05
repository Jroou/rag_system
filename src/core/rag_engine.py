from collections.abc import AsyncIterator

from src.generation.generator import Generator
from src.retrieval.base import RetrievalResult
from src.retrieval.reranker import Reranker
from src.routing.router import Router


class RAGEngine:
    def __init__(
        self,
        router: Router,
        reranker: Reranker,
        generator: Generator,
        top_k: int = 20,
        rerank_top_n: int = 5,
    ):
        self._router = router
        self._reranker = reranker
        self._generator = generator
        self._top_k = top_k
        self._rerank_top_n = rerank_top_n
        self._chat_history: list[dict] = []

    def query(self, user_query: str) -> tuple[str, list[RetrievalResult], str]:
        strategy_name, strategy = self._router.route(user_query)
        results = strategy.retrieve(user_query, top_k=self._top_k)

        if results:
            results = self._reranker.rerank(user_query, results, top_n=self._rerank_top_n)

        if not results:
            return "No relevant documents found.", [], strategy_name

        answer = self._generator.generate(
            user_query, results, chat_history=self._chat_history
        )

        self._chat_history.append({"role": "user", "content": user_query})
        self._chat_history.append({"role": "assistant", "content": answer})
        self._trim_history()

        return answer, results, strategy_name

    async def aquery_stream(
        self, user_query: str, strategy_override: str | None = None
    ) -> tuple[AsyncIterator[str], list[RetrievalResult], str]:
        if strategy_override and strategy_override != "auto":
            strategy = self._router.get_strategy(strategy_override)
            strategy_name = strategy_override
        else:
            strategy_name, strategy = self._router.route(user_query)
        results = strategy.retrieve(user_query, top_k=self._top_k)

        if results:
            results = self._reranker.rerank(user_query, results, top_n=self._rerank_top_n)

        if not results:

            async def empty():
                yield "No relevant documents found."

            return empty(), [], strategy_name

        try:
            stream = self._generator.agenerate_stream(
                user_query, results, chat_history=self._chat_history
            )
            # Attempt to get the first token to verify LLM availability
            first_token = await stream.__anext__()
        except Exception:
            return self._fallback_stream(results), results, strategy_name

        async def stream_with_first():
            yield first_token
            async for token in stream:
                yield token

        self._chat_history.append({"role": "user", "content": user_query})

        return stream_with_first(), results, strategy_name

    def _fallback_stream(
        self, results: list[RetrievalResult]
    ) -> AsyncIterator[str]:
        async def _stream():
            yield "⚠️ **LLM unavailable, showing raw results**\n\n"
            for i, r in enumerate(results, 1):
                text = r.parent_text or r.text
                yield f"**[Source {i}: {r.source_path}]**\n{text}\n\n---\n\n"

        return _stream()

    def record_response(self, response_text: str) -> None:
        self._chat_history.append({"role": "assistant", "content": response_text})
        self._trim_history()

    def update_generator(self, generator: Generator) -> None:
        self._generator = generator

    def _trim_history(self, max_messages: int = 20) -> None:
        if len(self._chat_history) > max_messages:
            self._chat_history = self._chat_history[-max_messages:]

    def clear_history(self) -> None:
        self._chat_history = []
