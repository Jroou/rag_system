import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv

load_dotenv()

import chainlit as cl

from src.core.config import load_config, save_config
from src.core.llm_factory import create_llm, create_llm_from_profile
from src.core.rag_engine import RAGEngine
from src.generation.generator import Generator
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.watcher import FileWatcher
from src.retrieval.hybrid import HybridStrategy
from src.retrieval.hyde import HyDEStrategy
from src.retrieval.reranker import Reranker
from src.retrieval.semantic import SemanticStrategy
from src.retrieval.stepback import StepBackStrategy
from src.routing.router import Router, STRATEGY_NAMES
from src.storage.qdrant_store import QdrantStore
from src.storage.sqlite_data_layer import SQLiteDataLayer
from src.storage.sqlite_store import SQLiteStore
from src.ui.command_dispatcher import CommandDispatcher

_config: dict[str, Any] | None = None
_qdrant: QdrantStore | None = None
_sqlite: SQLiteStore | None = None
_embedder: Embedder | None = None
_pipeline: IngestionPipeline | None = None
_engine: RAGEngine | None = None
_watcher: FileWatcher | None = None

def _get_llm_from_profile(profile: dict[str, Any]) -> Any:
    return create_llm_from_profile(profile)


def _get_llm(config: dict[str, Any]) -> Any:
    return create_llm(config)


def _build_strategies(qdrant: QdrantStore, embedder: Embedder, llm: Any) -> dict[str, Any]:
    semantic = SemanticStrategy(qdrant_store=qdrant, embedder=embedder)
    hybrid = HybridStrategy(qdrant_store=qdrant, embedder=embedder)
    hyde = HyDEStrategy(qdrant_store=qdrant, embedder=embedder, llm=llm)
    stepback = StepBackStrategy(qdrant_store=qdrant, embedder=embedder, llm=llm)
    return {
        "semantic": semantic,
        "hybrid": hybrid,
        "hyde": hyde,
        "stepback": stepback,
    }


def _initialize() -> None:
    global _config, _qdrant, _sqlite, _embedder, _pipeline, _engine, _watcher

    if _engine is not None:
        return

    _config = load_config()
    storage_cfg = _config["storage"]

    _embedder = Embedder(
        model_name=f"intfloat/{_config['embedding']['model_name']}",
        device=_config["embedding"]["device"],
    )
    _qdrant = QdrantStore(
        path=storage_cfg["qdrant_path"],
        collection_name=storage_cfg["collection_name"],
        vector_size=_embedder.vector_size,
    )
    _sqlite = SQLiteStore(db_path=storage_cfg["sqlite_path"])
    chunking_cfg = _config.get("chunking", {})
    _pipeline = IngestionPipeline(
        qdrant_store=_qdrant,
        sqlite_store=_sqlite,
        embedder=_embedder,
        parent_chunk_size=chunking_cfg.get("parent_chunk_size", 1000),
        child_chunk_size=chunking_cfg.get("child_chunk_size", 200),
        chunk_overlap=chunking_cfg.get("chunk_overlap", 50),
    )

    llm = _get_llm(_config)
    system_prompt = _config.get("system_prompt", "You are a helpful assistant.")
    generator = Generator(llm=llm, system_prompt=system_prompt)

    strategies = _build_strategies(_qdrant, _embedder, llm)
    router = Router(strategies=strategies)

    reranker_cfg = _config.get("reranker", {})
    reranker = Reranker(
        model_name=reranker_cfg.get("model_name", "BAAI/bge-reranker-v2-m3"),
        device=reranker_cfg.get("device", "cpu"),
    )

    _engine = RAGEngine(
        router=router,
        reranker=reranker,
        generator=generator,
        top_k=_config["retrieval"]["top_k"],
        rerank_top_n=reranker_cfg.get("top_n", 5),
    )

    kb_cfg = _config["knowledge_base"]
    _watcher = FileWatcher(
        folder=kb_cfg["monitored_folder"],
        supported_extensions=kb_cfg["supported_extensions"],
        on_ingest=_pipeline.ingest,
        on_delete=_pipeline.remove,
    )
    _watcher.start()


# Initialize at import time so models are loaded before Chainlit accepts connections.
# This avoids blocking the asyncio event loop on the first message and prevents
# WebSocket timeouts during the ~30s model load.
_initialize()


@cl.data_layer
def get_data_layer() -> SQLiteDataLayer:
    assert _sqlite is not None
    return SQLiteDataLayer(_sqlite)


# Auto-authenticate as a local user so Chainlit enables the thread history panel.
# Without an authenticated user the sidebar stays hidden regardless of data layer.
@cl.header_auth_callback
def header_auth_callback(headers: dict) -> cl.User | None:
    return cl.User(identifier="local", metadata={})


@cl.on_chat_resume
async def on_chat_resume(thread: dict) -> None:
    """Restore user session state when a user clicks a past thread in the sidebar."""
    _initialize()
    assert _config is not None
    thread_id = thread["id"]
    cl.user_session.set("profiles", list(_config["llm"]["profiles"].keys()))
    cl.user_session.set("active_profile", _config["llm"]["active_profile"])
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("strategy_override", "auto")


@cl.on_chat_start
async def on_chat_start() -> None:
    _initialize()
    assert _config is not None
    assert _sqlite is not None

    profiles = list(_config["llm"]["profiles"].keys())
    active = _config["llm"]["active_profile"]

    cl.user_session.set("profiles", profiles)
    cl.user_session.set("active_profile", active)

    # Use Chainlit's thread id (set by the data layer) or generate one
    thread_id = cl.context.session.thread_id or str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)

    # Register settings panel
    inputs: List[cl.input_widget.InputWidget] = [
            cl.input_widget.TextInput(
                id="system_prompt",
                label="System Prompt",
                initial=_config.get("system_prompt", "You are a helpful assistant."),
            ),
cl.input_widget.Select(
                id="profile",
                label="LLM Profile",
                values=profiles,
                initial_value=active,
            ),
            cl.input_widget.Slider(
                id="top_k",
                label="Top-K Retrieval",
                initial=_config["retrieval"]["top_k"],
                min=1,
                max=50,
                step=1,
            ),
            cl.input_widget.Slider(
                id="rerank_top_n",
                label="Rerank Top-N",
                initial=_config.get("reranker", {}).get("top_n", 5),
                min=1,
                max=20,
                step=1,
            ),
            cl.input_widget.Slider(
                id="temperature",
                label="Temperature",
                initial=_config["llm"]["profiles"][active].get("temperature", 0.3),
                min=0,
                max=1,
                step=0.05,
            ),
            cl.input_widget.Select(
                id="strategy_override",
                label="Strategy Override",
                values=["auto", *STRATEGY_NAMES],
                initial_value="auto",
            ),
    ]
    settings = await cl.ChatSettings(inputs).send()

    await cl.Message(
        content=f"Knowledge Base ready. Strategy routing and reranking enabled.\n"
        f"Active LLM Profile: **{active}**"
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    assert _config is not None
    assert _engine is not None

    changed = []

    # System prompt
    system_prompt = settings.get("system_prompt")
    if system_prompt and system_prompt != _config.get("system_prompt"):
        _config["system_prompt"] = system_prompt
        generator = Generator(llm=_engine._generator._llm, system_prompt=system_prompt)
        _engine.update_settings(generator=generator)
        changed.append("system prompt")

    # LLM Profile
    profile_name = settings.get("profile")
    if profile_name and profile_name in _config["llm"]["profiles"]:
        current = cl.user_session.get("active_profile")
        if profile_name != current:
            profile = _config["llm"]["profiles"][profile_name]
            llm = _get_llm_from_profile(profile)
            sp = _config.get("system_prompt", "You are a helpful assistant.")
            generator = Generator(llm=llm, system_prompt=sp)
            _engine.update_generator(generator)
            cl.user_session.set("active_profile", profile_name)
            changed.append(f"profile → {profile_name}")

    # Temperature
    temperature = settings.get("temperature")
    if temperature is not None:
        active_profile = cl.user_session.get("active_profile") or _config["llm"]["active_profile"]
        _config["llm"]["profiles"][active_profile]["temperature"] = temperature
        changed.append(f"temperature → {temperature}")

    # Top-K
    top_k = settings.get("top_k")
    if top_k is not None:
        _config["retrieval"]["top_k"] = int(top_k)
        _engine.update_settings(top_k=int(top_k))
        changed.append(f"top-K → {int(top_k)}")

    # Rerank Top-N
    rerank_top_n = settings.get("rerank_top_n")
    if rerank_top_n is not None:
        _config.setdefault("reranker", {})["top_n"] = int(rerank_top_n)
        _engine.update_settings(rerank_top_n=int(rerank_top_n))
        changed.append(f"rerank top-N → {int(rerank_top_n)}")

    # Strategy override
    strategy = settings.get("strategy_override")
    if strategy:
        cl.user_session.set("strategy_override", strategy)
        changed.append(f"strategy → {strategy}")

    if changed:
        save_config(_config)
        await cl.Message(content=f"Settings updated: {', '.join(changed)}").send()


async def _summarize_thread(thread_id: str) -> None:
    assert _sqlite is not None
    assert _engine is not None

    messages = _sqlite.get_thread_messages(thread_id)
    if not messages:
        return

    conversation = "\n".join(
        f"{m['role']}: {m['content'][:300]}" for m in messages[:20]
    )

    prompt = (
        "Summarize this conversation in 1-2 concise sentences capturing the key topics discussed:\n\n"
        f"{conversation}"
    )
    summary = await asyncio.to_thread(_engine.complete, prompt)
    if not summary:
        summary = messages[0]["content"][:50] if messages else "Empty thread"
    _sqlite.update_thread_summary(thread_id, summary)



def _readable_source_name(source_path: str, sqlite=None) -> str:
    if sqlite:
        doc = sqlite.get_document_by_path(source_path)
        if doc and doc.get("original_name"):
            return doc["original_name"]
    name = Path(source_path).name
    # Strip leading UUID (8-4-4-4-12 hex chars) followed by any separator
    cleaned = re.sub(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}[-_]', '', name, flags=re.IGNORECASE)
    return cleaned if cleaned != name else name


def _build_footnotes(results: list, cited_indices: set[int] | None = None, sqlite=None) -> str:
    seen_paths: set[str] = set()
    lines = ["\n\n---\n**Sources:**"]
    idx = 1
    for r in results:
        if cited_indices is None or idx in cited_indices:
            if r.source_path not in seen_paths:
                seen_paths.add(r.source_path)
                name = _readable_source_name(r.source_path, sqlite)
                lines.append(f"[{idx}] {name}")
        idx += 1
    return "\n".join(lines) if len(lines) > 1 else ""


@cl.on_message
async def on_message(message: cl.Message) -> None:
    global _engine, _pipeline, _config, _watcher

    if not _engine:
        _initialize()

    assert _engine is not None
    assert _pipeline is not None
    assert _config is not None

    thread_id = cl.user_session.get("thread_id")

    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path:
                file_path = Path(element.path)
                upload_name = getattr(element, "name", None) or file_path.name
                await cl.Message(content=f"Indexing: {upload_name}...").send()
                # Pass None for original_name so the pipeline can extract the real
                # title from PDF metadata; fall back to the upload filename.
                doc_id = await asyncio.to_thread(
                    _pipeline.ingest, file_path, False, None, thread_id
                )
                if doc_id:
                    # Read back the title the pipeline resolved
                    doc = _sqlite.get_document_by_path(str(file_path.resolve())) if _sqlite else None
                    display_name = (doc or {}).get("original_name") or upload_name
                    confirm_msg = cl.Message(content=f"Indexed: {display_name}")
                    if thread_id:
                        confirm_msg.actions = [
                            cl.Action(
                                name="promote_to_kb",
                                payload={"document_id": doc_id, "original_name": display_name},
                                label="📚 Add to knowledge base",
                            )
                        ]
                    await confirm_msg.send()
                else:
                    await cl.Message(content=f"Failed to index: {upload_name}").send()

    if not message.content.strip():
        return

    content = message.content.strip()

    # Command routing
    dispatcher = CommandDispatcher(_engine, _pipeline, _config, _sqlite, _watcher)
    if await dispatcher.dispatch(content):
        return

    # Normal query
    strategy_override = cl.user_session.get("strategy_override") or "auto"
    stream, results, _ = await _engine.aquery_stream(message.content, strategy_override=strategy_override, thread_id=thread_id)

    msg = cl.Message(content="")
    full_response = ""

    async for token in stream:
        full_response += token
        await msg.stream_token(token)

    if results:
        cited_indices = {int(m) for m in re.findall(r'\[(\d+)\]', full_response)}
        footnotes = _build_footnotes(results, cited_indices or None, sqlite=_sqlite)
        if footnotes:
            full_response += footnotes
            await msg.stream_token(footnotes)

    # Add "Save Finding" action if we have results
    if results and not full_response.startswith("⚠️"):
        actions = [
            cl.Action(
                name="save_finding",
                payload={"response": full_response, "results_json": json.dumps([
                    {"source_path": r.source_path, "text": (r.parent_text or r.text)[:200]}
                    for r in results
                ])},
                label="\U0001f4be Save Finding",
            )
        ]
        msg.actions = actions

    await msg.update()


@cl.action_callback("save_finding")
async def on_save_finding(action: cl.Action) -> None:
    assert _sqlite is not None
    assert _engine is not None

    payload = action.payload
    response_text = payload["response"]
    results_data = json.loads(payload["results_json"])

    citations = [{"source": r["source_path"], "text": r["text"]} for r in results_data]

    compress_prompt = (
        f"Compress the following answer into a single concise sentence "
        f"that captures the key fact. Include source references in brackets.\n\n"
        f"Answer: {response_text[:2000]}"
    )
    compressed = await asyncio.to_thread(_engine.complete, compress_prompt)
    if not compressed:
        compressed = response_text[:200] + ("..." if len(response_text) > 200 else "")

    finding_id = str(uuid.uuid4())
    _sqlite.add_finding(finding_id, compressed, json.dumps(citations))
    await cl.Message(content=f"💾 Finding saved: {compressed}").send()



@cl.on_stop
def on_stop() -> None:
    if _watcher:
        _watcher.stop()


@cl.action_callback("promote_to_kb")
async def on_promote_to_kb(action: cl.Action) -> None:
    document_id = action.payload.get("document_id")
    original_name = action.payload.get("original_name", "document")
    if document_id and _pipeline:
        await asyncio.to_thread(_pipeline.promote_to_global, document_id)
        await cl.Message(content=f"📚 **{original_name}** added to knowledge base.").send()

