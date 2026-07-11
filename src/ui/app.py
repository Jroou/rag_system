import asyncio
import json
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
from src.routing.router import Router
from src.storage.qdrant_store import QdrantStore
from src.storage.sqlite_store import SQLiteStore
from src.ui.command_dispatcher import CommandDispatcher

_config: dict[str, Any] | None = None
_qdrant: QdrantStore | None = None
_sqlite: SQLiteStore | None = None
_embedder: Embedder | None = None
_pipeline: IngestionPipeline | None = None
_engine: RAGEngine | None = None
_watcher: FileWatcher | None = None
_inactivity_tasks: dict[str, asyncio.Task] = {}


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

    _qdrant = QdrantStore(
        path=storage_cfg["qdrant_path"],
        collection_name=storage_cfg["collection_name"],
    )
    _sqlite = SQLiteStore(db_path=storage_cfg["sqlite_path"])
    _embedder = Embedder(
        model_name=f"intfloat/{_config['embedding']['model_name']}",
        device=_config["embedding"]["device"],
    )
    _pipeline = IngestionPipeline(
        qdrant_store=_qdrant,
        sqlite_store=_sqlite,
        embedder=_embedder,
        config=_config,
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


@cl.on_chat_start
async def on_chat_start() -> None:
    _initialize()
    assert _config is not None
    assert _sqlite is not None

    profiles = list(_config["llm"]["profiles"].keys())
    active = _config["llm"]["active_profile"]

    cl.user_session.set("profiles", profiles)
    cl.user_session.set("active_profile", active)

    # Create a new thread for this session
    thread_id = str(uuid.uuid4())
    _sqlite.create_thread(thread_id, "New conversation")
    cl.user_session.set("thread_id", thread_id)

    # Register settings panel
    inputs: List[cl.input_widget.InputWidget] = [
            cl.input_widget.TextInput(
                id="system_prompt",
                label="System Prompt",
                initial=_config.get("system_prompt", "You are a helpful assistant."),
            ),
            cl.input_widget.Slider(
                id="inactivity_timeout",
                label="Inactivity Timeout (minutes)",
                initial=_config.get("inactivity_timeout", 30),
                min=1,
                max=180,
                step=1,
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
                values=["auto", "semantic", "hybrid", "hyde", "stepback"],
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
        new_generator = Generator(
            llm=create_llm(_config), system_prompt=system_prompt
        )
        _engine.update_settings(generator=new_generator)
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

    # Inactivity timeout
    timeout = settings.get("inactivity_timeout")
    if timeout is not None:
        _config["inactivity_timeout"] = int(timeout)
        changed.append(f"inactivity timeout → {int(timeout)} min")

    # Strategy override
    strategy = settings.get("strategy_override")
    if strategy:
        cl.user_session.set("strategy_override", strategy)
        changed.append(f"strategy → {strategy}")

    if changed:
        save_config(_config)
        await cl.Message(content=f"Settings updated: {', '.join(changed)}").send()


async def _handle_save_finding(response_text: str, results: list) -> None:
    assert _engine is not None
    assert _sqlite is not None

    citations = []
    for r in results:
        citations.append({"source": r.source_path, "text": (r.parent_text or r.text)[:200]})

    try:
        llm = _engine._generator._llm
        compress_prompt = (
            f"Compress the following answer into a single concise sentence "
            f"that captures the key fact. Include source references in brackets.\n\n"
            f"Answer: {response_text[:2000]}"
        )
        resp = llm.invoke([{"role": "user", "content": compress_prompt}])
        compressed = resp.content
    except Exception:
        compressed = response_text[:200] + ("..." if len(response_text) > 200 else "")

    finding_id = str(uuid.uuid4())
    _sqlite.add_finding(finding_id, compressed, json.dumps(citations))
    await cl.Message(content=f"💾 Finding saved: {compressed}").send()


async def _handle_findings_list() -> None:
    assert _sqlite is not None
    findings = _sqlite.list_findings()
    if not findings:
        await cl.Message(content="No saved findings.").send()
        return

    lines = ["## Saved Findings\n"]
    for f in findings:
        citations = json.loads(f["citations"])
        cite_str = ", ".join(Path(c["source"]).name for c in citations) if citations else "no sources"
        lines.append(f"- **{f['text']}** ({cite_str}) `[id: {f['id'][:8]}]`")
    await cl.Message(content="\n".join(lines)).send()


async def _handle_delete_finding(finding_id_prefix: str) -> None:
    assert _sqlite is not None
    findings = _sqlite.list_findings()
    match = [f for f in findings if f["id"].startswith(finding_id_prefix)]
    if match:
        _sqlite.delete_finding(match[0]["id"])
        await cl.Message(content=f"Deleted finding `{match[0]['id'][:8]}`").send()
    else:
        await cl.Message(content="Finding not found.").send()


async def _handle_settings_show() -> None:
    assert _config is not None
    import yaml

    await cl.Message(content=f"```yaml\n{yaml.dump(_config, default_flow_style=False)}\n```").send()


async def _handle_settings_set(key_path: str, value: str) -> None:
    global _config, _engine, _watcher, _pipeline
    assert _config is not None

    keys = key_path.split(".")
    obj = _config
    for k in keys[:-1]:
        if k not in obj:
            await cl.Message(content=f"Invalid key: {key_path}").send()
            return
        obj = obj[k]

    try:
        parsed_value = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        parsed_value = value

    obj[keys[-1]] = parsed_value
    save_config(_config)

    if key_path == "knowledge_base.monitored_folder" and _watcher and _pipeline:
        _watcher.stop()
        kb_cfg = _config["knowledge_base"]
        _watcher = FileWatcher(
            folder=kb_cfg["monitored_folder"],
            supported_extensions=kb_cfg["supported_extensions"],
            on_ingest=_pipeline.ingest,
            on_delete=_pipeline.remove,
        )
        _watcher.start()

    if key_path == "system_prompt" and _engine:
        llm = _engine._generator._llm
        generator = Generator(llm=llm, system_prompt=parsed_value)
        _engine.update_generator(generator)

    await cl.Message(content=f"Setting `{key_path}` updated.").send()


async def _handle_documents_list() -> None:
    assert _sqlite is not None
    docs = _sqlite.list_documents()
    if not docs:
        await cl.Message(content="No indexed documents.").send()
        return

    lines = ["## Indexed Documents\n"]
    lines.append("| Name | Type | Status | Last Indexed |")
    lines.append("|------|------|--------|--------------|")
    for d in docs:
        name = Path(d["source_path"]).name
        error = f" ⚠️ {d['error_message']}" if d.get("error_message") else ""
        lines.append(f"| {name} | {d['document_type']} | {d['status']}{error} | {d['indexed_at'][:19]} |")
    await cl.Message(content="\n".join(lines)).send()


async def _handle_reindex(file_name: str) -> None:
    assert _sqlite is not None
    assert _pipeline is not None
    docs = _sqlite.list_documents()
    match = [d for d in docs if Path(d["source_path"]).name == file_name or d["source_path"] == file_name]
    if not match:
        await cl.Message(content=f"Document `{file_name}` not found in index.").send()
        return

    path = Path(match[0]["source_path"])
    await cl.Message(content=f"Re-indexing: {path.name}...").send()
    doc_id = await asyncio.to_thread(_pipeline.ingest, path, force=True)
    if doc_id:
        await cl.Message(content=f"Re-indexed: {path.name}").send()
    else:
        await cl.Message(content=f"Failed to re-index: {path.name}").send()


async def _handle_history() -> None:
    assert _sqlite is not None
    threads = _sqlite.list_threads(limit=20)
    if not threads:
        await cl.Message(content="No conversation history.").send()
        return

    lines = ["## Conversation History\n"]
    for t in threads:
        display = t["summary"] or t["title"] or "Untitled"
        date = t["updated_at"][:16].replace("T", " ")
        lines.append(f"- **{display}** — {date} `[{t['id'][:8]}]`")
    await cl.Message(content="\n".join(lines)).send()


async def _handle_doc_delete(file_name: str) -> None:
    assert _sqlite is not None
    assert _pipeline is not None
    docs = _sqlite.list_documents()
    match = [d for d in docs if Path(d["source_path"]).name == file_name or d["source_path"] == file_name]
    if not match:
        await cl.Message(content=f"Document `{file_name}` not found in index.").send()
        return

    path = Path(match[0]["source_path"])
    await asyncio.to_thread(_pipeline.remove, path)
    await cl.Message(content=f"Deleted from index: {path.name}").send()


async def _summarize_thread(thread_id: str) -> None:
    assert _sqlite is not None
    assert _engine is not None

    messages = _sqlite.get_thread_messages(thread_id)
    if not messages:
        return

    conversation = "\n".join(
        f"{m['role']}: {m['content'][:300]}" for m in messages[:20]
    )

    try:
        llm = _engine._generator._llm
        prompt = (
            "Summarize this conversation in 1-2 concise sentences capturing the key topics discussed:\n\n"
            f"{conversation}"
        )
        resp = await asyncio.to_thread(
            llm.invoke, [{"role": "user", "content": prompt}]
        )
        summary = resp.content if isinstance(resp.content, str) else str(resp.content)
        _sqlite.update_thread_summary(thread_id, summary)
    except Exception:
        first_msg = messages[0]["content"][:50] if messages else "Empty thread"
        _sqlite.update_thread_summary(thread_id, first_msg)


async def _inactivity_watcher(thread_id: str) -> None:
    assert _config is not None
    timeout_minutes = _config.get("inactivity_timeout", 30)
    await asyncio.sleep(timeout_minutes * 60)
    await _summarize_thread(thread_id)


def _reset_inactivity_timer(thread_id: str) -> None:
    if thread_id in _inactivity_tasks:
        _inactivity_tasks[thread_id].cancel()
    _inactivity_tasks[thread_id] = asyncio.create_task(_inactivity_watcher(thread_id))


def _build_footnotes(results: list) -> str:
    seen_paths: set[str] = set()
    lines = ["\n\n---\n**Sources:**"]
    idx = 1
    for r in results:
        if r.source_path not in seen_paths:
            seen_paths.add(r.source_path)
            name = Path(r.source_path).name
            excerpt = (r.parent_text or r.text)[:100].replace("\n", " ")
            lines.append(f"[{idx}] {name} — \"{excerpt}...\"")
            idx += 1
    return "\n".join(lines) if idx > 1 else ""


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
                await cl.Message(content=f"Indexing: {file_path.name}...").send()
                doc_id = await asyncio.to_thread(_pipeline.ingest, file_path)
                if doc_id:
                    await cl.Message(content=f"Indexed: {file_path.name}").send()
                else:
                    await cl.Message(content=f"Failed to index: {file_path.name}").send()

    if not message.content.strip():
        return

    content = message.content.strip()

    # Persist user message to thread
    if thread_id and _sqlite:
        _sqlite.add_thread_message(str(uuid.uuid4()), thread_id, "user", content)
        # Update title from first real message
        thread = _sqlite.get_thread(thread_id)
        if thread and thread["title"] == "New conversation":
            _sqlite.update_thread_title(thread_id, content[:50])
        _reset_inactivity_timer(thread_id)

    # Command routing
    dispatcher = CommandDispatcher(_engine, _pipeline, _config, _sqlite, _watcher)
    if await dispatcher.dispatch(content):
        return

    # Normal query
    strategy_override = cl.user_session.get("strategy_override") or "auto"
    stream, results, _ = await _engine.aquery_stream(message.content, strategy_override=strategy_override)

    msg = cl.Message(content="")
    full_response = ""

    async for token in stream:
        full_response += token
        await msg.stream_token(token)

    if results:
        # Append footnote block with source citations
        footnotes = _build_footnotes(results)
        if footnotes:
            full_response += footnotes
            await msg.stream_token(footnotes)

        sources = []
        seen_paths: set[str] = set()
        for r in results:
            if r.source_path not in seen_paths:
                seen_paths.add(r.source_path)
                source_text = r.parent_text or r.text
                sources.append(
                    cl.Text(
                        name=Path(r.source_path).name,
                        content=source_text,
                        display="side",
                    )
                )
        msg.elements = sources

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
    _engine.record_response(full_response)

    # Persist assistant response to thread
    if thread_id and _sqlite:
        _sqlite.add_thread_message(str(uuid.uuid4()), thread_id, "assistant", full_response)


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
