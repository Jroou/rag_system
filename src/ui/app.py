import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import chainlit as cl

from src.core.config import load_config, save_config
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

_config: dict[str, Any] | None = None
_qdrant: QdrantStore | None = None
_sqlite: SQLiteStore | None = None
_embedder: Embedder | None = None
_pipeline: IngestionPipeline | None = None
_engine: RAGEngine | None = None
_watcher: FileWatcher | None = None


def _get_llm_from_profile(profile: dict[str, Any]) -> Any:
    provider = profile["provider"]
    model = profile["model"]
    temperature = profile.get("temperature", 0.3)
    api_key_env = profile.get("api_key_env", "")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, temperature=temperature, anthropic_api_key=api_key  # type: ignore[call-arg]
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)  # type: ignore[call-arg]
    elif provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(
            model_id=model,  # type: ignore[call-arg]
            temperature=temperature,
            region_name=profile.get("region", "us-east-1"),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _get_llm(config: dict[str, Any]) -> Any:
    profile_name = config["llm"]["active_profile"]
    profile = config["llm"]["profiles"][profile_name]
    return _get_llm_from_profile(profile)


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

    profiles = list(_config["llm"]["profiles"].keys())
    active = _config["llm"]["active_profile"]

    chat_profiles = [
        cl.ChatProfile(name=name, markdown_description=f"LLM Profile: {name}")
        for name in profiles
    ]
    if chat_profiles:
        cl.user_session.set("profiles", profiles)
        cl.user_session.set("active_profile", active)

    await cl.Message(
        content=f"Knowledge Base ready. Strategy routing and reranking enabled.\n"
        f"Active LLM Profile: **{active}**"
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    assert _config is not None
    assert _engine is not None

    profile_name = settings.get("profile")
    if profile_name and profile_name in _config["llm"]["profiles"]:
        profile = _config["llm"]["profiles"][profile_name]
        llm = _get_llm_from_profile(profile)
        system_prompt = _config.get("system_prompt", "You are a helpful assistant.")
        generator = Generator(llm=llm, system_prompt=system_prompt)
        _engine.update_generator(generator)
        cl.user_session.set("active_profile", profile_name)
        await cl.Message(content=f"Switched to profile: **{profile_name}**").send()


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

    if key_path == "knowledge_base.monitored_folder" and _watcher:
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


@cl.on_message
async def on_message(message: cl.Message) -> None:
    global _engine, _pipeline, _config, _watcher

    if not _engine:
        _initialize()

    assert _engine is not None
    assert _pipeline is not None
    assert _config is not None

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

    # Command routing
    if content.startswith("/profile "):
        profile_name = content[len("/profile "):].strip()
        if profile_name in _config["llm"]["profiles"]:
            profile = _config["llm"]["profiles"][profile_name]
            llm = _get_llm_from_profile(profile)
            system_prompt = _config.get("system_prompt", "You are a helpful assistant.")
            generator = Generator(llm=llm, system_prompt=system_prompt)
            _engine.update_generator(generator)
            await cl.Message(content=f"Switched to profile: **{profile_name}**").send()
        else:
            available = ", ".join(_config["llm"]["profiles"].keys())
            await cl.Message(content=f"Unknown profile. Available: {available}").send()
        return

    if content == "/findings":
        await _handle_findings_list()
        return
    if content.startswith("/finding delete "):
        await _handle_delete_finding(content[len("/finding delete "):].strip())
        return

    if content == "/settings":
        await _handle_settings_show()
        return
    if content.startswith("/settings set "):
        parts = content[len("/settings set "):].strip().split(" ", 1)
        if len(parts) == 2:
            await _handle_settings_set(parts[0], parts[1])
        else:
            await cl.Message(content="Usage: `/settings set <key.path> <value>`").send()
        return

    if content == "/documents":
        await _handle_documents_list()
        return
    if content.startswith("/reindex "):
        await _handle_reindex(content[len("/reindex "):].strip())
        return
    if content.startswith("/doc delete "):
        await _handle_doc_delete(content[len("/doc delete "):].strip())
        return

    # Normal query
    stream, results, _ = await _engine.aquery_stream(message.content)

    msg = cl.Message(content="")
    full_response = ""

    async for token in stream:
        full_response += token
        await msg.stream_token(token)

    if results:
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
                label="💾 Save Finding",
            )
        ]
        msg.actions = actions

    await msg.update()
    _engine.record_response(full_response)


@cl.action_callback("save_finding")
async def on_save_finding(action: cl.Action) -> None:
    assert _sqlite is not None
    assert _engine is not None

    payload = action.payload
    response_text = payload["response"]
    results_data = json.loads(payload["results_json"])

    citations = [{"source": r["source_path"], "text": r["text"]} for r in results_data]

    try:
        llm = _engine._generator._llm
        compress_prompt = (
            f"Compress the following answer into a single concise sentence "
            f"that captures the key fact. Include source references in brackets.\n\n"
            f"Answer: {response_text[:2000]}"
        )
        resp = await asyncio.to_thread(
            llm.invoke, [{"role": "user", "content": compress_prompt}]
        )
        compressed = resp.content
    except Exception:
        compressed = response_text[:200] + ("..." if len(response_text) > 200 else "")

    finding_id = str(uuid.uuid4())
    _sqlite.add_finding(finding_id, compressed, json.dumps(citations))
    await cl.Message(content=f"💾 Finding saved: {compressed}").send()


@cl.on_stop
def on_stop() -> None:
    if _watcher:
        _watcher.stop()
