# Adaptive retrieval with rule-based routing

The system automatically selects a retrieval Strategy (semantic, hybrid, HyDE, step-back) based on Query characteristics, using a heuristic classifier with a lightweight ML model fallback. We chose this over agentic RAG (LLM-as-orchestrator) because it avoids an extra LLM call per query (+1-3s latency, additional cost) while still providing adaptive behavior. We chose it over static/manual selection because users shouldn't need to understand retrieval internals to get good results.

## Considered Options

- **Agentic RAG**: LLM decides retrieval strategy per query. More flexible, but adds latency and cost on every query — unacceptable for a local personal tool where responsiveness matters.
- **Manual selection**: User picks strategy via dropdown. Requires domain knowledge from the user, adds friction.
- **Single strategy (semantic only)**: Simplest, but conceptual queries get poor results without HyDE, and keyword-heavy queries miss without hybrid search.