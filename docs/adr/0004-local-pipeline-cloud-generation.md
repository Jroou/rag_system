# Local pipeline with cloud LLM for generation

All pipeline components (embedding, chunking, retrieval, reranking, routing) run locally. Only the final generation step calls a cloud LLM API. This preserves document privacy at the storage level (nothing leaves the machine until a query is made), while getting significantly better generation quality than local LLMs can currently provide for RAG synthesis tasks. Multiple providers are supported via LangChain's BaseChatModel abstraction, configured as switchable Profiles.

When the cloud API is unavailable, the system degrades gracefully: retrieval still works and raw Chunks with citations are shown to the user, only the synthesized answer is unavailable.