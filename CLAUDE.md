# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues on this repo (`Jroou/rag_system`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — one `CONTEXT.md` + `docs/adr/` at the root. See `docs/agents/domain.md`.

---

## Commands

```bash
# Install dependencies (uv required)
uv sync

# Run the app (starts Chainlit UI on port 8000)
uv run start
# or directly:
uv run python -m chainlit run src/ui/app.py --host 0.0.0.0 --port 8000

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_router.py

# Run a single test by name
uv run pytest tests/test_router.py::test_classify_query_hybrid

# Eval pipeline (requires golden dataset indexed first)
uv run python tests/eval/index_golden_ds.py   # one-time indexing
uv run python tests/eval/run_eval.py          # retrieval eval
uv run python tests/eval/score_pipeline.py    # LLM scoring pass
```

## Architecture

### Data flow

Query → **Router** → **Strategy** → **Reranker** → **Generator** → streamed response

Ingestion: file drop/watcher event → **IngestionPipeline** → type-aware chunker → **Embedder** → Qdrant + SQLite

### Core components

**`src/core/rag_engine.py`** — central orchestrator. Holds a `Router`, `Reranker`, `Generator`, and in-memory chat history (trimmed to 20 messages). `aquery_stream()` is the main entry point; it peeks at the first token to detect LLM failures and falls back to raw chunk display.

**`src/routing/router.py`** — rule-based query classifier. Dispatches to one of four strategies: `semantic` (short queries), `hybrid` (keyword/error queries), `hyde` (long conceptual), `stepback` (narrow/specific). Patterns are bilingual (Ukrainian + English).

**`src/ingestion/pipeline.py`** — deduplicates by file hash (SQLite), deletes stale Qdrant vectors on re-ingest, then produces parent+child chunks. Only child chunks are used for search; parent chunks are fetched at generation time for fuller context.

**`src/storage/`** — two stores. Qdrant (embedded, `./qdrant_data`) holds vectors. SQLite (`./rag_system.db`) holds document metadata, conversation threads, messages, and findings.

**`src/ui/app.py`** — Chainlit app. All global state (`_engine`, `_pipeline`, `_watcher`, etc.) is module-level singletons initialized once on first chat. Supports in-chat slash commands: `/profile`, `/documents`, `/reindex`, `/doc delete`, `/findings`, `/finding delete`, `/settings`, `/settings set`, `/history`, `/compress`.

**`src/core/llm_factory.py`** — creates LangChain chat models from `config/settings.yaml` profiles. Supports `anthropic`, `openai`, and `bedrock` providers.

### Chunking strategy (ADR-0003)

Documents are split into **parent chunks** (≈1000 tokens, for generation context) and **child chunks** (≈200 tokens, for search). Both are stored as Qdrant vectors. Retrieval matches child chunks, then the pipeline resolves and injects their parent text into the generation prompt.

### Storage layout

- `./qdrant_data/` — embedded Qdrant vector store (committed collection, not gitignored)
- `./rag_system.db` — SQLite: documents, threads, messages, findings
- `./data/` — monitored folder; drop documents here for auto-ingestion
- `config/settings.yaml` — all tuneable knobs (LLM profiles, chunking sizes, retrieval top-k, reranker model)

### Eval infrastructure

`tests/eval/` contains a standalone evaluation harness separate from pytest. It uses its own Qdrant instance (`tests/eval/qdrant_eval_data/`) and golden dataset (`tests/eval/golden_queries.json`). Source PDFs live in `~/Documents/rag-golden-ds/` (not committed). `score_pipeline.py` calls the LLM to judge completeness and formatting against `key_facts` defined in the golden queries.

## Domain vocabulary

Use precise terms from `CONTEXT.md`: **Chunk** (not fragment), **Strategy** (not method/mode), **Router** (not dispatcher), **Finding** (not note/bookmark), **Thread** (not chat/session), **Ingestion** (not indexing). The codebase and user-facing strings follow this vocabulary consistently.
