from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FusionQuery,
    IsNullCondition,
    MatchValue,
    PayloadField,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

_UNSET = object()


def _thread_condition(thread_id: str | None):
    """Return a Qdrant condition that matches vectors for the given thread_id.

    thread_id=<uuid>  → match that thread's documents
    thread_id=None    → match global KB documents (thread_id field is null/absent)
    """
    if thread_id is not None:
        return FieldCondition(key="thread_id", match=MatchValue(value=thread_id))
    return IsNullCondition(is_null=PayloadField(key="thread_id"))


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
            points.append(PointStruct(id=point_id, vector=vectors, payload=metadata))
        self._client.upsert(collection_name=self._collection_name, points=points)

    def _make_filter(
        self,
        filter_conditions: dict | None = None,
        thread_id=_UNSET,
    ) -> Filter | None:
        must = []
        if filter_conditions:
            for key, value in filter_conditions.items():
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        if thread_id is not _UNSET:
            must.append(_thread_condition(thread_id))
        return Filter(must=must) if must else None

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        filter_conditions: dict | None = None,
        thread_id=_UNSET,
    ) -> list[dict]:
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            using="dense",
            limit=top_k,
            query_filter=self._make_filter(filter_conditions, thread_id),
            with_payload=True,
        )
        return [
            {"id": point.id, "score": point.score, "metadata": point.payload}
            for point in results.points
        ]

    def search_with_fallback(
        self,
        query_embedding: list[float],
        top_k: int,
        thread_id: str,
        filter_conditions: dict | None = None,
        fallback_threshold: int = 2,
    ) -> list[dict]:
        """Search thread-scoped docs first; fall back to global KB if results are sparse."""
        thread_results = self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_conditions=filter_conditions,
            thread_id=thread_id,
        )
        if len(thread_results) >= fallback_threshold:
            return thread_results
        global_results = self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_conditions=filter_conditions,
            thread_id=None,
        )
        seen = {r["id"] for r in thread_results}
        merged = thread_results + [r for r in global_results if r["id"] not in seen]
        return merged[:top_k]

    def fetch_parent(self, parent_chunk_id: str) -> str | None:
        try:
            points, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="chunk_type", match=MatchValue(value="parent"))]
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
        thread_id=_UNSET,
    ) -> list[dict]:
        child_filter = self._make_filter({"chunk_type": "child"}, thread_id)

        prefetch = [
            Prefetch(query=query_embedding, using="dense", limit=top_k * 2, filter=child_filter),
        ]
        if sparse_vector is not None:
            prefetch.append(
                Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2, filter=child_filter)
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

    def search_rrf_with_fallback(
        self,
        query_embedding: list[float],
        top_k: int,
        thread_id: str,
        sparse_vector: SparseVector | None = None,
        fallback_threshold: int = 2,
    ) -> list[dict]:
        """RRF search with thread-first, global-KB fallback."""
        thread_results = self.search_rrf(
            query_embedding=query_embedding,
            top_k=top_k,
            sparse_vector=sparse_vector,
            thread_id=thread_id,
        )
        if len(thread_results) >= fallback_threshold:
            return thread_results
        global_results = self.search_rrf(
            query_embedding=query_embedding,
            top_k=top_k,
            sparse_vector=sparse_vector,
            thread_id=None,
        )
        seen = {r["id"] for r in thread_results}
        merged = thread_results + [r for r in global_results if r["id"] not in seen]
        return merged[:top_k]

    def delete_by_document_id(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )

    def count(self) -> int:
        info = self._client.get_collection(self._collection_name)
        return info.points_count

    def close(self) -> None:
        self._client.close()
