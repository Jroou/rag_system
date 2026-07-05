# RAG System

Local personal knowledge base with adaptive retrieval — a system that indexes personal documents and answers questions by automatically selecting the most appropriate search strategy.

## Language

### Core Concepts

**Knowledge Base**:
A persistent, locally-stored collection of indexed documents available for retrieval.
_Avoid_: corpus, database, library

**Document**:
A single source file (PDF, Markdown, DOC, or code file) that has been ingested into the Knowledge Base.
_Avoid_: file, resource, asset

**Chunk**:
A segment of a Document, produced by a type-aware splitter, stored as an embedding in the vector store.
_Avoid_: fragment, segment, passage, block

**Parent Chunk**:
A larger contextual segment that contains one or more Child Chunks. Used for generation context after retrieval matches a Child Chunk.
_Avoid_: context window, surrounding text

**Child Chunk**:
A small, precise segment used for embedding-based search. Links back to its Parent Chunk.
_Avoid_: sub-chunk, micro-chunk

**Finding**:
A user-saved, LLM-compressed key fact extracted from a conversation, with citations to source Documents.
_Avoid_: note, summary, bookmark, snippet

### Retrieval

**Strategy**:
A specific retrieval approach (semantic, hybrid, HyDE, step-back) that defines how a Query is transformed and matched against Chunks.
_Avoid_: method, technique, algorithm, mode

**Router**:
The component that classifies a Query and selects the appropriate Strategy. Uses rule-based heuristics with a lightweight ML classifier fallback.
_Avoid_: dispatcher, selector, orchestrator

**Query**:
A user's natural-language question submitted to the system for retrieval and generation.
_Avoid_: prompt, request, question

**Reranker**:
A cross-encoder model that re-scores retrieved Chunks for relevance after initial retrieval, before passing to the LLM.
_Avoid_: scorer, sorter, filter

### Ingestion

**Ingestion**:
The process of loading, chunking, embedding, and storing a Document into the Knowledge Base.
_Avoid_: indexing, processing, importing

**Watcher**:
A file-system listener that detects new/changed/deleted Documents in the monitored folder and queues them for Ingestion.
_Avoid_: monitor, observer, listener

**Debounce**:
A 1-2 second delay after the last file-system event before Ingestion begins, ensuring the file is fully written.
_Avoid_: cooldown, delay, throttle

### Generation

**Profile**:
A named LLM configuration (provider, model, temperature) that the user can switch between in the UI.
_Avoid_: preset, config, mode

**Citation**:
An inline numbered reference (e.g., [1]) in a generated response that links a specific claim to a source Document. A footnote block at the end maps numbers to Document names.
_Avoid_: reference, link, source tag

### Conversation

**Thread**:
A single conversation session between the user and the system, containing an ordered sequence of messages. Persisted in SQLite across restarts.
_Avoid_: chat, session, conversation

**Summary**:
An LLM-compressed one-sentence description of a Thread's content, generated either by inactivity timeout or manual user action. Displayed in the history sidebar.
_Avoid_: digest, recap, title

**Inactivity Timeout**:
The configurable duration (default 30 minutes) after the last message in a Thread before automatic summarization triggers.
_Avoid_: idle timer, session expiry, cooldown
