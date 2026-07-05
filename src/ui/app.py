import os
from pathlib import Path

import chainlit as cl

from src.core.config import load_config
from src.core.rag_engine import RAGEngine
from src.generation.generator import Generator
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.semantic import SemanticStrategy
from src.storage.qdrant_store import QdrantStore
from src.storage.sqlite_store import SQLiteStore

_config = None
_qdrant = None
_sqlite = None
_embedder = None
_pipeline = None
_engine = None


def _get_llm(config: dict):
    profile_name = config["llm"]["active_profile"]
    profile = config["llm"]["profiles"][profile_name]
    provider = profile["provider"]
    model = profile["model"]
    temperature = profile.get("temperature", 0.3)
    api_key_env = profile.get("api_key_env", "")
    api_key = os.environ.get(api_key_env, "")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, temperature=temperature, anthropic_api_key=api_key
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _initialize():
    global _config, _qdrant, _sqlite, _embedder, _pipeline, _engine

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
    strategy = SemanticStrategy(qdrant_store=_qdrant, embedder=_embedder)

    _engine = RAGEngine(
        strategy=strategy,
        generator=generator,
        top_k=_config["retrieval"]["top_k"],
    )


@cl.on_chat_start
async def on_chat_start():
    _initialize()
    await cl.Message(content="Knowledge Base ready. Ask me anything about your documents.").send()


@cl.on_message
async def on_message(message: cl.Message):
    if not _engine:
        _initialize()

    # Handle file uploads
    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path:
                file_path = Path(element.path)
                doc_id = _pipeline.ingest(file_path)
                if doc_id:
                    await cl.Message(
                        content=f"Indexed: {file_path.name}"
                    ).send()
                else:
                    await cl.Message(
                        content=f"Failed to index: {file_path.name}"
                    ).send()

    if not message.content.strip():
        return

    # Stream response
    stream, results = await _engine.aquery_stream(message.content)

    msg = cl.Message(content="")
    full_response = ""

    async for token in stream:
        full_response += token
        await msg.stream_token(token)

    # Add citations
    if results:
        sources = []
        seen_paths = set()
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

    await msg.send()
    _engine.record_response(full_response)
