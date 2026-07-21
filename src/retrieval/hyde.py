from src.retrieval.base import BaseStrategy, RetrievalResult
from src.retrieval.semantic import SemanticStrategy
from src.storage.qdrant_store import QdrantStore

HYDE_PROMPT = """Given the following question, write a short paragraph that would be a plausible answer found in a document. Do not explain that you are generating a hypothetical answer — just write the paragraph directly.

Question: {query}

Hypothetical answer:"""


class HyDEStrategy(BaseStrategy):
    def __init__(self, qdrant_store: QdrantStore, embedder, llm):
        self._qdrant = qdrant_store
        self._embedder = embedder
        self._llm = llm
        self._fallback = SemanticStrategy(qdrant_store=qdrant_store, embedder=embedder)

    def retrieve(self, query: str, top_k: int = 20, thread_id: str | None = None, document_ids: list[str] | None = None) -> list[RetrievalResult]:
        try:
            hypothetical_answer = self._generate_hypothesis(query)
        except Exception:
            return self._fallback.retrieve(query, top_k, thread_id=thread_id, document_ids=document_ids)

        hyp_embedding = self._embedder.embed_query(hypothetical_answer)

        if thread_id is not None:
            child_results = self._qdrant.search_with_fallback(
                query_embedding=hyp_embedding,
                top_k=top_k,
                thread_id=thread_id,
                filter_conditions={"chunk_type": "child"},
                document_ids=document_ids,
            )
        else:
            child_results = self._qdrant.search(
                query_embedding=hyp_embedding,
                top_k=top_k,
                filter_conditions={"chunk_type": "child"},
                document_ids=document_ids,
            )

        results = []
        seen_parents = set()

        for hit in child_results:
            metadata = hit["metadata"]
            parent_chunk_id = metadata.get("parent_chunk_id")
            parent_text = None

            if parent_chunk_id and parent_chunk_id not in seen_parents:
                seen_parents.add(parent_chunk_id)
                parent_text = self._qdrant.fetch_parent(parent_chunk_id)

            results.append(
                RetrievalResult(
                    chunk_id=hit["id"],
                    text=metadata.get("text", ""),
                    score=hit["score"],
                    source_path=metadata.get("source_path", ""),
                    document_type=metadata.get("document_type", ""),
                    parent_text=parent_text,
                    document_id=metadata.get("document_id"),
                )
            )

        return results

    def _generate_hypothesis(self, query: str) -> str:
        prompt = HYDE_PROMPT.format(query=query)
        response = self._llm.invoke([{"role": "user", "content": prompt}])
        return response.content
