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

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        try:
            hypothetical_answer = self._generate_hypothesis(query)
        except Exception:
            return self._fallback.retrieve(query, top_k)

        hyp_embedding = self._embedder.embed_query(hypothetical_answer)

        child_results = self._qdrant.search(
            query_embedding=hyp_embedding,
            top_k=top_k,
            filter_conditions={"chunk_type": "child"},
        )

        results = []
        seen_parents = set()

        for hit in child_results:
            metadata = hit["metadata"]
            parent_chunk_id = metadata.get("parent_chunk_id")
            parent_text = None

            if parent_chunk_id and parent_chunk_id not in seen_parents:
                seen_parents.add(parent_chunk_id)
                parent_text = self._resolve_parent_text(parent_chunk_id)

            results.append(
                RetrievalResult(
                    chunk_id=hit["id"],
                    text=metadata.get("text", ""),
                    score=hit["score"],
                    source_path=metadata.get("source_path", ""),
                    document_type=metadata.get("document_type", ""),
                    parent_text=parent_text,
                )
            )

        return results

    def _generate_hypothesis(self, query: str) -> str:
        prompt = HYDE_PROMPT.format(query=query)
        response = self._llm.invoke([{"role": "user", "content": prompt}])
        return response.content

    def _resolve_parent_text(self, parent_chunk_id: str) -> str | None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        try:
            points = self._qdrant._client.scroll(
                collection_name=self._qdrant._collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="chunk_type", match=MatchValue(value="parent")),
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False,
            )[0]
            for point in points:
                if str(point.id) == parent_chunk_id:
                    return point.payload.get("text")
        except Exception:
            pass
        return None
