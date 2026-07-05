from qdrant_client.models import (
    FusionQuery,
    Prefetch,
    QueryRequest,
    SparseVector,
)

from src.retrieval.base import BaseStrategy, RetrievalResult
from src.storage.qdrant_store import QdrantStore


class HybridStrategy(BaseStrategy):
    def __init__(self, qdrant_store: QdrantStore, embedder, sparse_embedder=None):
        self._qdrant = qdrant_store
        self._embedder = embedder
        self._sparse_embedder = sparse_embedder

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        query_embedding = self._embedder.embed_query(query)

        prefetch = [
            Prefetch(
                query=query_embedding,
                using="dense",
                limit=top_k * 2,
            ),
        ]

        if self._sparse_embedder:
            sparse_vector = self._sparse_embedder.embed_query_sparse(query)
            prefetch.append(
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=top_k * 2,
                )
            )

        results = self._qdrant._client.query_points(
            collection_name=self._qdrant._collection_name,
            prefetch=prefetch,
            query=FusionQuery(fusion="rrf"),
            limit=top_k,
            with_payload=True,
            query_filter={
                "must": [{"key": "chunk_type", "match": {"value": "child"}}]
            },
        )

        retrieval_results = []
        seen_parents = set()

        for hit in results.points:
            metadata = hit.payload
            parent_chunk_id = metadata.get("parent_chunk_id")
            parent_text = None

            if parent_chunk_id and parent_chunk_id not in seen_parents:
                seen_parents.add(parent_chunk_id)
                parent_text = self._resolve_parent_text(parent_chunk_id)

            retrieval_results.append(
                RetrievalResult(
                    chunk_id=str(hit.id),
                    text=metadata.get("text", ""),
                    score=hit.score,
                    source_path=metadata.get("source_path", ""),
                    document_type=metadata.get("document_type", ""),
                    parent_text=parent_text,
                )
            )

        return retrieval_results

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
