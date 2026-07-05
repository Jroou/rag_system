from collections.abc import AsyncIterator

from src.generation.generator import Generator
from src.retrieval.base import BaseStrategy, RetrievalResult


class RAGEngine:
    def __init__(
        self,
        strategy: BaseStrategy,
        generator: Generator,
        top_k: int = 20,
    ):
        self._strategy = strategy
        self._generator = generator
        self._top_k = top_k
        self._chat_history: list[dict] = []

    def query(self, user_query: str) -> tuple[str, list[RetrievalResult]]:
        results = self._strategy.retrieve(user_query, top_k=self._top_k)
        if not results:
            return "No relevant documents found.", []

        answer = self._generator.generate(
            user_query, results, chat_history=self._chat_history
        )

        self._chat_history.append({"role": "user", "content": user_query})
        self._chat_history.append({"role": "assistant", "content": answer})
        self._trim_history()

        return answer, results

    async def aquery_stream(
        self, user_query: str
    ) -> tuple[AsyncIterator[str], list[RetrievalResult]]:
        results = self._strategy.retrieve(user_query, top_k=self._top_k)
        if not results:

            async def empty():
                yield "No relevant documents found."

            return empty(), []

        stream = self._generator.agenerate_stream(
            user_query, results, chat_history=self._chat_history
        )

        self._chat_history.append({"role": "user", "content": user_query})

        return stream, results

    def record_response(self, response_text: str) -> None:
        self._chat_history.append({"role": "assistant", "content": response_text})
        self._trim_history()

    def _trim_history(self, max_messages: int = 20) -> None:
        if len(self._chat_history) > max_messages:
            self._chat_history = self._chat_history[-max_messages:]

    def clear_history(self) -> None:
        self._chat_history = []
