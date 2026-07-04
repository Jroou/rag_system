Status: ready-for-agent

# Modular RAG System — Personal Knowledge Base

## Problem Statement

A user has a growing collection of personal documents (PDFs, Markdown notes, code files, DOC files) spread across their local machine. Finding specific information across these documents requires manually opening and searching each file. Existing solutions either send all data to the cloud (privacy concern), require complex infrastructure setup, or use a one-size-fits-all retrieval approach that performs poorly on diverse query types.

## Solution

A locally-running personal Knowledge Base with an adaptive retrieval pipeline that automatically selects the best search Strategy based on the Query characteristics. Documents are ingested from a monitored folder (with UI upload as secondary input), chunked using document-type-aware splitters with parent-child linking, and stored in a local vector database. A rule-based Router classifies incoming Queries and routes them to the appropriate Strategy (semantic, hybrid, HyDE, or step-back). Results are reranked locally, then passed to a configurable cloud LLM for answer synthesis with source citations. The system exposes a Chainlit chat UI with streaming, clickable citations, LLM Profile switching, and a Findings save mechanism.

## User Stories

1. As a user, I want to point the system at a folder of my documents, so that they are automatically indexed and searchable.
2. As a user, I want to drop a new file into the monitored folder, so that it becomes searchable within seconds without manual action.
3. As a user, I want to delete a file from the folder, so that its content is removed from search results automatically.
4. As a user, I want to see indexing progress in the UI ("Indexing 2/5..."), so that I know what the system is doing.
5. As a user, I want to upload a file directly through the UI, so that I can add documents without navigating to the folder.
6. As a user, I want to ask a natural-language question, so that the system retrieves relevant information from my documents.
7. As a user, I want the system to automatically choose the best retrieval Strategy for my Query, so that I get the best results without understanding retrieval internals.
8. As a user, I want to see the answer streamed token-by-token, so that I don't stare at a blank screen while waiting.
9. As a user, I want clickable source citations on each answer, so that I can verify the information and read the original context.
10. As a user, I want to click a citation and see the relevant chunk from the source Document, so that I can understand the context around the answer.
11. As a user, I want to save a key Finding from a conversation, so that I can recall important facts later without re-asking.
12. As a user, I want to see my saved Findings with their citations, so that I can review what I've learned from my documents.
13. As a user, I want to switch between LLM Profiles (e.g., fast/cheap vs. quality), so that I can balance speed and cost per query.
14. As a user, I want to edit the system prompt, so that I can customize the response style and behavior.
15. As a user, I want to edit configuration through the UI (monitored folder path, LLM profiles, embedding settings), so that I don't need to edit YAML files manually.
16. As a user, I want the system to still show me retrieved Chunks with citations when the cloud LLM is unavailable, so that I can still find information.
17. As a user, I want to see a list of all indexed Documents with their status, so that I know what's in my Knowledge Base.
18. As a user, I want to manually trigger a re-index of a specific Document, so that I can force refresh after edits.
19. As a user, I want to start the entire system with a single command (`uv run start`), so that I don't need to manage multiple processes.
20. As a user, I want the system to work with Ukrainian and English documents equally well, so that my multilingual notes are all searchable.
21. As a user, I want PDFs to be chunked respecting their structural elements (headers, paragraphs), so that retrieved chunks make sense in isolation.
22. As a user, I want code files to be chunked by functions/classes, so that I get meaningful code snippets in search results.
23. As a user, I want Markdown files to be chunked by headers, so that each chunk represents a coherent section.

## Implementation Decisions

### Architecture

- **Pipeline**: LangChain as the orchestration abstraction across all pipeline stages.
- **Vector Store**: Qdrant in embedded mode (in-process Python). Supports hybrid search (sparse + dense) and metadata filtering natively. Configurable to connect to an external Qdrant instance if needed.
- **Embedding Model**: `multilingual-e5-large` running locally on CPU. Handles Ukrainian + English content.
- **Reranker**: `BGE-reranker-v2-m3` running locally on CPU. Applied after initial retrieval to re-score top-K candidates before generation.
- **LLM**: Cloud API via LangChain `BaseChatModel`. Multiple providers supported (Anthropic, OpenAI, etc.) through configurable Profiles.
- **UI**: Chainlit — provides streaming, citations, file upload, chat interface out of the box.

### Retrieval Strategies

Four initial Strategies:
- **Semantic**: Direct embedding similarity search. For factual, specific queries.
- **Hybrid**: BM25 + semantic (Qdrant native). For queries with important specific terms.
- **HyDE**: LLM generates hypothetical answer → embed that → search. For abstract/conceptual queries.
- **Step-back**: LLM generates a broader question → search for general context → answer specific question. For overly narrow queries.

### Routing

- **Primary**: Rule-based heuristic classifier (query length, question words, presence of specific terms, query language).
- **Fallback**: Lightweight ML classifier (e.g., fasttext/sklearn) trained on accumulated query data.
- Default Strategy when uncertain: heuristic-based selection (semantic as safest default).

### Chunking

- Document-type-aware splitters:
  - PDF: paragraph/header-based splitting
  - Markdown: `MarkdownHeaderTextSplitter`
  - Code: Language-aware splitter (by function/class)
  - DOC/DOCX: paragraph-based splitting
- Parent-child linking: Child Chunks (~200 tokens) for search precision, Parent Chunks (~1000 tokens) for generation context.
- LangChain `ParentDocumentRetriever` as the retrieval interface.

### Ingestion & File Watching

- **Watcher**: `watchdog` library monitoring a configured folder.
- **Debounce**: 1-2 second delay after last modification event before queueing Ingestion.
- **Task Queue**: `queue.Queue` with a separate worker thread. Watcher adds paths to queue; worker processes them. UI remains responsive.
- **Deletion sync**: File removed from folder → corresponding vectors removed from Qdrant.
- **On-demand fallback**: Manual "Reindex" button for files changed while the app was stopped.
- **Change detection**: File hashes (stored in SQLite) to avoid re-indexing unchanged files.

### Storage & Configuration

- **Config**: YAML file (`config/settings.yaml`). Editable via UI and directly. Contains: monitored folder paths, LLM Profiles, embedding model settings, routing rules, system prompt.
- **Operational data**: SQLite database. Contains: document metadata (path, hash, status, timestamps), Findings (compressed facts + citations), ingestion queue state.
- **Vector data**: Qdrant embedded storage (managed by Qdrant itself).

### Chat & Memory

- **In-session**: Token-based window (fill available LLM context window with conversation history).
- **Cross-session**: No full chat history persistence. Instead, user saves Findings on demand — LLM compresses the answer into a one-liner with source citations, stored in SQLite.

### Error Handling

- **Graceful degradation**: When cloud LLM is unavailable, retrieval pipeline continues working. UI shows retrieved Chunks with citations and a message "LLM unavailable, showing raw results".
- No fallback to local LLM — quality gap is too large for RAG synthesis.

### Project Structure

```
rag_system/
├── config/
│   └── settings.yaml
├── src/
│   ├── core/              # shared interfaces, base classes
│   ├── ingestion/         # file watcher, document loaders, chunking
│   ├── retrieval/         # strategies (semantic, hybrid, hyde, step-back)
│   ├── routing/           # query classifier, strategy selector
│   ├── generation/        # LLM interaction, prompt templates
│   ├── storage/           # Qdrant client, SQLite, findings store
│   └── ui/                # Chainlit app, settings panel
├── tests/
├── data/                  # default monitored folder
├── pyproject.toml
└── CONTEXT.md
```

### Runtime

- Python 3.12+
- Dependency management: `uv`
- Single entrypoint: `uv run start` launches Qdrant embedded + Chainlit app in one process
- CPU-only inference for embedding and reranking (~8GB RAM minimum, 16GB comfortable)

## Testing Decisions

Tests verify external behavior at module boundaries — not internal implementation details.

### Seams

1. **Ingestion pipeline**: Given a file path + type → produces correct Parent/Child Chunks with expected metadata. Tests document loaders and chunkers independently per document type.
2. **Router**: Given a Query string → returns the correct Strategy name. Tests heuristic rules and classifier without touching retrieval.
3. **Retrieval strategies**: Given a Query + fixture collection in Qdrant → returns ranked results matching expected documents. Each Strategy tested independently.
4. **Reranker**: Given a Query + candidate Chunks → reordered list matches expected ranking.
5. **Generation**: Given Chunks + Query + Profile config → correct prompt assembly. Mock LLM call, verify prompt structure.
6. **File Watcher**: Given filesystem events → correct queue entries after debounce. Tests debounce logic and deletion signaling without actual ingestion.
7. **Config/Storage**: Given YAML config changes → system reflects new state. Tests config loading, Profile switching, Finding persistence.

### Approach

- Integration tests against real Qdrant embedded instance (not mocked — ADR-0002 chose embedded specifically for testability).
- Unit tests for Router heuristics, debounce logic, chunking output.
- Fixture documents (small PDF, MD, code, DOC samples) in test data directory.

## Out of Scope

- Multi-user support or authentication
- Remote/cloud deployment
- Full conversation history persistence (replaced by Findings mechanism)
- Agentic RAG (LLM-as-orchestrator for retrieval)
- GPU acceleration
- OCR for scanned PDFs or images
- Web scraping / URL ingestion
- Real-time collaboration
- Mobile UI
- Automatic Finding extraction (user-triggered only)

## Further Notes

### Phased Delivery

**MVP (Phase 1):**
- Ingestion pipeline (PDF, MD, code, DOC with document-type-aware chunking + parent-child)
- Qdrant embedded + multilingual-e5-large embeddings
- Single retrieval Strategy (semantic) for a working end-to-end system
- Chainlit UI with streaming, citations, file upload
- YAML config
- Single entrypoint `uv run start`

**Phase 2:**
- Hybrid search, HyDE, step-back Strategies
- Rule-based Router + lightweight classifier fallback
- Reranker (BGE-reranker-v2-m3)
- File Watcher with debounce + task queue
- LLM Profiles (switching in UI)

**Phase 3:**
- Findings store (save key findings on demand)
- UI settings panel (system prompt, config editing)
- Document management panel (index status, deletion, re-indexing)
- Graceful degradation when LLM unavailable