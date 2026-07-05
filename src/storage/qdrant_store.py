from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

DENSE_VECTOR_SIZE = 1024  # multilingual-e5-large


class QdrantStore:
    def __init__(self, path: str, collection_name: str):
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._client = QdrantClient(path=str(self._path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection_name not in collections:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=DENSE_VECTOR_SIZE,
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
