from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large", device: str = "cpu"):
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = self._model.encode(prefixed, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        prefixed = f"query: {query}"
        embedding = self._model.encode([prefixed], normalize_embeddings=True)
        return embedding[0].tolist()
