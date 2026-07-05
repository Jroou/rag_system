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
        self, user_query: str
    ) -> tuple[AsyncIterator[str], list[RetrievalResult], str]:
        strategy_name, strategy = self._router.route(user_query)
        results = strategy.retrieve(user_query, top_k=self._top_k)

        if results:
            results = self._reranker.rerank(user_query, results, top_n=self._rerank_top_n)

        if not results:

            async def empty():
                yield "No relevant documents found."

            return empty(), [], strategy_name

        stream = self._generator.agenerate_stream(
            user_query, results, chat_history=self._chat_history
        )

        self._chat_history.append({"role": "user", "content": user_query})

        return stream, results, strategy_name

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
