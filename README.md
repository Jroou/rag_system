# RAG System

A local personal knowledge base with adaptive retrieval. Drop documents in, ask questions, get cited answers.

Built as a PySide6 desktop app with a fully offline vector pipeline — no data leaves your machine unless you configure a cloud LLM.

## Features

- **Adaptive retrieval** — rule-based router selects the best strategy per query (semantic, hybrid, HyDE, step-back)
- **Parent-child chunking** — small chunks for precise search, large chunks for rich generation context
- **Document-scoped queries** — check specific documents to restrict retrieval scope
- **Multi-chat threads** — each thread can have its own scoped documents with global KB fallback
- **Live ingestion** — drag-and-drop or file watcher with stage progress and cancellation
- **Cross-encoder reranking** — BGE reranker-v2 for precision after recall
- **Citation grounding** — every claim links to a numbered source

## Quick start

```bash
# Requires Python 3.12+ and uv
uv sync
uv run start
```

The app opens a desktop window. Drop PDF/DOCX/MD files into the document panel or place them in `./data/` for auto-ingestion.

## Configuration

All settings live in `config/settings.yaml`:

| Section | Controls |
|---------|----------|
| `llm` | Provider profiles (Anthropic, OpenAI, Bedrock), active profile, temperature |
| `embedding` | Model name, device (cpu/cuda) |
| `chunking` | Parent/child chunk sizes, overlap |
| `retrieval` | Default strategy, top-k |
| `reranker` | Model, device, top-n |
| `knowledge_base` | Monitored folder, supported extensions |

Create a `.env` file for API keys:

```
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
```

## Architecture

```
Query → Router → Strategy → Reranker → Generator → streamed response

Ingestion: file → type-aware chunker → Embedder → Qdrant + SQLite
```

### Key components

| Component | Path | Role |
|-----------|------|------|
| RAG Engine | `src/core/rag_engine.py` | Orchestrates retrieval → rerank → generation |
| Router | `src/routing/router.py` | Classifies queries into strategies (bilingual UA+EN) |
| Strategies | `src/retrieval/` | Semantic, Hybrid (RRF), HyDE, Step-back |
| Pipeline | `src/ingestion/pipeline.py` | Dedup, chunk, embed, store with progress reporting |
| Qdrant Store | `src/storage/qdrant_store.py` | Embedded vector DB with thread-scoped + global search |
| SQLite Store | `src/storage/sqlite_store.py` | Document metadata, threads, messages |
| Generator | `src/generation/generator.py` | Prompt construction with citation instructions |
| UI | `src/ui/` | PySide6 desktop app with async bridge |

### Storage

- `./qdrant_data/` — embedded Qdrant vector store
- `./rag_system.db` — SQLite (documents, threads, messages)
- `./data/` — monitored folder for auto-ingestion

## Testing

```bash
uv run pytest                    # all tests
uv run pytest tests/test_router.py  # single file
```

### Eval pipeline

A separate evaluation harness lives in `tests/eval/`:

```bash
uv run python tests/eval/index_golden_ds.py   # index golden dataset
uv run python tests/eval/run_eval.py          # retrieval eval
uv run python tests/eval/score_pipeline.py    # LLM scoring
```

## Design decisions

Architectural decisions are recorded in `docs/adr/`:

- **ADR-0001** — Adaptive retrieval with rule-based routing
- **ADR-0002** — Qdrant embedded as vector store
- **ADR-0003** — Parent-child chunking strategy
- **ADR-0004** — Local pipeline, cloud generation
