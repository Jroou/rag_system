# Qdrant embedded as vector store

We use Qdrant in embedded mode (in-process Python) as the default vector store. This gives zero-setup single-process deployment while providing native hybrid search (sparse + dense vectors) and metadata filtering needed by the Router. We chose Qdrant over ChromaDB because ChromaDB lacks hybrid search and has weaker filtering. We chose embedded over Docker deployment because a personal knowledge base doesn't need multi-process architecture — one `uv run start` should bring everything up.

## Considered Options

- **ChromaDB**: Simpler API, but no native hybrid search and limited metadata filtering — would require manual BM25 implementation alongside.
- **FAISS**: Fast but no persistence or filtering out of the box — prototype-only.
- **Qdrant via Docker**: More scalable, but requires Docker and multi-process management for no real benefit at personal KB scale (<100k vectors).