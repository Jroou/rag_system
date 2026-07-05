from sentence_transformers import CrossEncoder

from src.retrieval.base import RetrievalResult


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        self._model = CrossEncoder(model_name, device=device)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_n: int = 5
    ) -> list[RetrievalResult]:
        if not results:
            return []

        pairs = [(query, r.parent_text or r.text) for r in results]
        scores = self._model.predict(pairs)

        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for result, score in scored[:top_n]:
            reranked.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=float(score),
                    source_path=result.source_path,
                    document_type=result.document_type,
                    parent_text=result.parent_text,
                )
            )
        return reranked
