from src.retrieval.base import BaseStrategy, RetrievalResult
from src.storage.qdrant_store import QdrantStore


class SemanticStrategy(BaseStrategy):
    def __init__(self, qdrant_store: QdrantStore, embedder):
        self._qdrant = qdrant_store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        query_embedding = self._embedder.embed_query(query)

        child_results = self._qdrant.search(
            query_embedding=query_embedding,
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
                parent_text = self._qdrant.fetch_parent(parent_chunk_id)

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
