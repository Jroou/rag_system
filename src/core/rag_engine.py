from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from src.generation.generator import Generator
from src.retrieval.base import RetrievalResult
from src.retrieval.reranker import Reranker
from src.routing.router import Router

if TYPE_CHECKING:
    from src.storage.qdrant_store import QdrantStore


def _dedup_by_document(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Keep only the highest-ranked chunk per document, preserving rerank order."""
    seen: set[str] = set()
    out = []
    for r in results:
        key = r.document_id or r.source_path
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


class RAGEngine:
    def __init__(
        self,
        router: Router,
        reranker: Reranker,
        generator: Generator,
        top_k: int = 20,
        rerank_top_n: int = 5,
        qdrant_store: QdrantStore | None = None,
    ):
        self._router = router
        self._reranker = reranker
        self._generator = generator
        self._top_k = top_k
        self._rerank_top_n = rerank_top_n
        self._qdrant = qdrant_store
        self._chat_history: list[dict] = []

    def query(self, user_query: str) -> tuple[str, list[RetrievalResult], str]:
        strategy_name, strategy = self._router.route(user_query)
        results = strategy.retrieve(user_query, top_k=self._top_k)

        if results:
            results = self._reranker.rerank(user_query, results, top_n=self._rerank_top_n)
            results = _dedup_by_document(results)

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
        self, user_query: str, strategy_override: str | None = None, thread_id: str | None = None, document_ids: list[str] | None = None
    ) -> tuple[AsyncIterator[str], list[RetrievalResult], str]:
        if strategy_override and strategy_override != "auto":
            strategy = self._router.get_strategy(strategy_override)
            strategy_name = strategy_override
        else:
            strategy_name, strategy = self._router.route(user_query)
        results = strategy.retrieve(user_query, top_k=self._top_k, thread_id=thread_id, document_ids=document_ids)

        if results:
            results = self._reranker.rerank(user_query, results, top_n=self._rerank_top_n)
            results = _dedup_by_document(results)

        if document_ids and self._qdrant:
            results = self._prepend_doc_context(results, document_ids)

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

        self._chat_history.append({"role": "user", "content": user_query})

        async def stream_with_history():
            accumulated = first_token
            yield first_token
            async for token in stream:
                accumulated += token
                yield token
            self._chat_history.append({"role": "assistant", "content": accumulated})
            self._trim_history()

        return stream_with_history(), results, strategy_name

    def _fallback_stream(
        self, results: list[RetrievalResult]
    ) -> AsyncIterator[str]:
        async def _stream():
            yield "⚠️ **LLM unavailable, showing raw results**\n\n"
            for i, r in enumerate(results, 1):
                text = r.parent_text or r.text
                yield f"**[Source {i}: {r.source_path}]**\n{text}\n\n---\n\n"

        return _stream()

    def complete(self, prompt: str) -> str:
        try:
            response = self._generator._llm.invoke([{"role": "user", "content": prompt}])
            return response.content
        except Exception:
            return ""

    def update_settings(
        self,
        *,
        top_k: int | None = None,
        rerank_top_n: int | None = None,
        generator: Generator | None = None,
    ) -> None:
        if top_k is not None:
            self._top_k = top_k
        if rerank_top_n is not None:
            self._rerank_top_n = rerank_top_n
        if generator is not None:
            self._generator = generator

    def update_generator(self, generator: Generator) -> None:
        self._generator = generator

    def _prepend_doc_context(
        self, results: list[RetrievalResult], document_ids: list[str]
    ) -> list[RetrievalResult]:
        """Prepend the first parent chunk (title/abstract) for scoped documents.

        Ensures the LLM always sees the document opening even when semantic
        search only matches later sections.
        """
        first_chunks = self._qdrant.fetch_first_parent_chunks(document_ids)
        existing_texts = {(r.parent_text or r.text) for r in results}

        prepended = []
        for doc_id in document_ids:
            entry = first_chunks.get(doc_id)
            if not entry:
                continue
            text = entry["text"]
            if text in existing_texts:
                continue
            prepended.append(
                RetrievalResult(
                    chunk_id=f"{doc_id}_intro",
                    text=text,
                    score=1.0,
                    source_path=entry["source_path"],
                    document_type="",
                    document_id=doc_id,
                    parent_text=None,
                )
            )
        return prepended + results

    def _trim_history(self, max_messages: int = 20) -> None:
        if len(self._chat_history) > max_messages:
            self._chat_history = self._chat_history[-max_messages:]

    def clear_history(self) -> None:
        self._chat_history = []
