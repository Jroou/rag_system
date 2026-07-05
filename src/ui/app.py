import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import chainlit as cl

from src.core.config import load_config
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


@cl.on_message
async def on_message(message: cl.Message) -> None:
    global _engine, _pipeline, _config

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

    try:
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

        await msg.update()
        _engine.record_response(full_response)
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()


@cl.on_stop
def on_stop() -> None:
    if _watcher:
        _watcher.stop()
