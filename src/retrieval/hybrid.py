from src.retrieval.base import BaseStrategy, RetrievalResult
from src.storage.qdrant_store import QdrantStore


class HybridStrategy(BaseStrategy):
    def __init__(self, qdrant_store: QdrantStore, embedder, sparse_embedder=None):
        self._qdrant = qdrant_store
        self._embedder = embedder
        self._sparse_embedder = sparse_embedder

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        query_embedding = self._embedder.embed_query(query)

        sparse_vector = None
        if self._sparse_embedder:
            sparse_vector = self._sparse_embedder.embed_query_sparse(query)

        hits = self._qdrant.search_rrf(query_embedding, top_k, sparse_vector=sparse_vector)

        retrieval_results = []
        seen_parents = set()

        for hit in hits:
            metadata = hit["metadata"]
            parent_chunk_id = metadata.get("parent_chunk_id")
            parent_text = None

            if parent_chunk_id and parent_chunk_id not in seen_parents:
                seen_parents.add(parent_chunk_id)
                parent_text = self._qdrant.fetch_parent(parent_chunk_id)

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=str(hit["id"]),
                    text=metadata.get("text", ""),
                    score=hit["score"],
                    source_path=metadata.get("source_path", ""),
                    document_type=metadata.get("document_type", ""),
                    parent_text=parent_text,
                )
            )

        return retrieval_results
