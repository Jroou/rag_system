from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

class QdrantStore:
    def __init__(self, path: str, collection_name: str, vector_size: int = 1024):
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._client = QdrantClient(path=str(self._path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection_name not in collections:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self._vector_size,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
            )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        sparse_vectors: list[SparseVector | None] | None = None,
    ) -> None:
        points = []
        for i, (point_id, embedding, metadata) in enumerate(
            zip(ids, embeddings, metadatas)
        ):
            vectors = {"dense": embedding}
            if sparse_vectors and sparse_vectors[i] is not None:
                vectors["sparse"] = sparse_vectors[i]

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload=metadata,
                )
            )
        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        filter_conditions: dict | None = None,
    ) -> list[dict]:
        qdrant_filter = None
        if filter_conditions:
            must = []
            for key, value in filter_conditions.items():
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))
            qdrant_filter = Filter(must=must)

        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            using="dense",
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            {"id": point.id, "score": point.score, "metadata": point.payload}
            for point in results.points
        ]

    def fetch_parent(self, parent_chunk_id: str) -> str | None:
        """Scroll Qdrant for a parent chunk by ID and return its text, or None if not found."""
        try:
            points, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="chunk_type", match=MatchValue(value="parent")),
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                if str(point.id) == parent_chunk_id:
                    return point.payload.get("text")
        except Exception:
            pass
        return None

    def search_rrf(
        self,
        query_embedding: list[float],
        top_k: int,
        sparse_vector: SparseVector | None = None,
    ) -> list[dict]:
        """Perform RRF fusion query using Qdrant's FusionQuery + Prefetch.

        Filters to chunk_type='child' and returns results in the same shape as
        search(): list of dicts with 'id', 'score', 'metadata' keys.
        """
        child_filter = Filter(
            must=[
                FieldCondition(key="chunk_type", match=MatchValue(value="child")),
            ]
        )

        prefetch = [
            Prefetch(
                query=query_embedding,
                using="dense",
                limit=top_k * 2,
                filter=child_filter,
            ),
        ]

        if sparse_vector is not None:
            prefetch.append(
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=top_k * 2,
                    filter=child_filter,
                )
            )

        results = self._client.query_points(
            collection_name=self._collection_name,
            prefetch=prefetch,
            query=FusionQuery(fusion="rrf"),
            limit=top_k,
            with_payload=True,
        )
        return [
            {"id": point.id, "score": point.score, "metadata": point.payload}
            for point in results.points
        ]

    def delete_by_document_id(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=document_id)
                    )
                ]
            ),
        )

    def count(self) -> int:
        info = self._client.get_collection(self._collection_name)
        return info.points_count

    def close(self) -> None:
        self._client.close()
