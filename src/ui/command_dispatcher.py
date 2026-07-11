"""CommandDispatcher — routes slash commands from on_message() to handlers.

Usage in app.py::

    dispatcher = CommandDispatcher(_engine, _pipeline, _config, _sqlite, _watcher)
    if await dispatcher.dispatch(content):
        return
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import chainlit as cl

from src.core.config import save_config
from src.core.llm_factory import create_llm, create_llm_from_profile
from src.generation.generator import Generator
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.watcher import FileWatcher


class CommandDispatcher:
    """Routes slash-command content strings to the appropriate async handler.

    Returns True from dispatch() when a command was consumed; False when the
    content is a plain query and the caller should proceed with RAG.
    """

    def __init__(
        self,
        engine: Any,
        pipeline: IngestionPipeline,
        config: dict[str, Any],
        sqlite: Any,
        watcher: FileWatcher | None = None,
    ) -> None:
        self._engine = engine
        self._pipeline = pipeline
        self._config = config
        self._sqlite = sqlite
        self._watcher = watcher

    async def dispatch(self, content: str) -> bool:
        """Return True if content matched a slash command, False otherwise."""
        if content.startswith("/profile "):
            await self._handle_profile(content[len("/profile "):].strip())
            return True

        if content == "/findings":
            await self._handle_findings()
            return True

        if content.startswith("/finding delete "):
            await self._handle_finding_delete(content[len("/finding delete "):].strip())
            return True

        if content == "/settings":
            await self._handle_settings_show()
            return True

        if content.startswith("/settings set "):
            parts = content[len("/settings set "):].strip().split(" ", 1)
            if len(parts) == 2:
                await self._handle_settings_set(parts[0], parts[1])
            else:
                await cl.Message(content="Usage: `/settings set <key.path> <value>`").send()
            return True

        if content == "/documents":
            await self._handle_documents()
            return True

        if content.startswith("/reindex "):
            await self._handle_reindex(content[len("/reindex "):].strip())
            return True

        if content.startswith("/doc delete "):
            await self._handle_doc_delete(content[len("/doc delete "):].strip())
            return True

        if content == "/history":
            await self._handle_history()
            return True

        if content == "/compress":
            await self._handle_compress()
            return True

        return False

    # ------------------------------------------------------------------
    # Private handlers
    # ------------------------------------------------------------------

    async def _handle_profile(self, profile_name: str) -> None:
        if profile_name in self._config["llm"]["profiles"]:
            profile = self._config["llm"]["profiles"][profile_name]
            llm = create_llm_from_profile(profile)
            system_prompt = self._config.get("system_prompt", "You are a helpful assistant.")
            generator = Generator(llm=llm, system_prompt=system_prompt)
            self._engine.update_generator(generator)
            await cl.Message(content=f"Switched to profile: **{profile_name}**").send()
        else:
            available = ", ".join(self._config["llm"]["profiles"].keys())
            await cl.Message(content=f"Unknown profile. Available: {available}").send()

    async def _handle_findings(self) -> None:
        findings = self._sqlite.list_findings()
        if not findings:
            await cl.Message(content="No saved findings.").send()
            return

        lines = ["## Saved Findings\n"]
        for f in findings:
            citations = json.loads(f["citations"])
            cite_str = (
                ", ".join(Path(c["source"]).name for c in citations)
                if citations
                else "no sources"
            )
            lines.append(f"- **{f['text']}** ({cite_str}) `[id: {f['id'][:8]}]`")
        await cl.Message(content="\n".join(lines)).send()

    async def _handle_finding_delete(self, finding_id_prefix: str) -> None:
        findings = self._sqlite.list_findings()
        match = [f for f in findings if f["id"].startswith(finding_id_prefix)]
        if match:
            self._sqlite.delete_finding(match[0]["id"])
            await cl.Message(content=f"Deleted finding `{match[0]['id'][:8]}`").send()
        else:
            await cl.Message(content="Finding not found.").send()

    async def _handle_settings_show(self) -> None:
        import yaml

        await cl.Message(
            content=f"```yaml\n{yaml.dump(self._config, default_flow_style=False)}\n```"
        ).send()

    async def _handle_settings_set(self, key_path: str, value: str) -> None:
        keys = key_path.split(".")
        obj = self._config
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
        save_config(self._config)

        if key_path == "knowledge_base.monitored_folder" and self._watcher and self._pipeline:
            self._watcher.stop()
            kb_cfg = self._config["knowledge_base"]
            self._watcher = FileWatcher(
                folder=kb_cfg["monitored_folder"],
                supported_extensions=kb_cfg["supported_extensions"],
                on_ingest=self._pipeline.ingest,
                on_delete=self._pipeline.remove,
            )
            self._watcher.start()

        if key_path == "system_prompt" and self._engine:
            llm = create_llm(self._config)
            generator = Generator(llm=llm, system_prompt=parsed_value)
            self._engine.update_settings(generator=generator)

        await cl.Message(content=f"Setting `{key_path}` updated.").send()

    async def _handle_documents(self) -> None:
        docs = self._sqlite.list_documents()
        if not docs:
            await cl.Message(content="No indexed documents.").send()
            return

        lines = ["## Indexed Documents\n"]
        lines.append("| Name | Type | Status | Last Indexed |")
        lines.append("|------|------|--------|--------------|")
        for d in docs:
            name = Path(d["source_path"]).name
            error = f" ⚠️ {d['error_message']}" if d.get("error_message") else ""
            lines.append(
                f"| {name} | {d['document_type']} | {d['status']}{error}"
                f" | {d['indexed_at'][:19]} |"
            )
        await cl.Message(content="\n".join(lines)).send()

    async def _handle_reindex(self, file_name: str) -> None:
        docs = self._sqlite.list_documents()
        match = [
            d
            for d in docs
            if Path(d["source_path"]).name == file_name or d["source_path"] == file_name
        ]
        if not match:
            await cl.Message(content=f"Document `{file_name}` not found in index.").send()
            return

        path = Path(match[0]["source_path"])
        await cl.Message(content=f"Re-indexing: {path.name}...").send()
        doc_id = await asyncio.to_thread(self._pipeline.ingest, path, force=True)
        if doc_id:
            await cl.Message(content=f"Re-indexed: {path.name}").send()
        else:
            await cl.Message(content=f"Failed to re-index: {path.name}").send()

    async def _handle_doc_delete(self, file_name: str) -> None:
        docs = self._sqlite.list_documents()
        match = [
            d
            for d in docs
            if Path(d["source_path"]).name == file_name or d["source_path"] == file_name
        ]
        if not match:
            await cl.Message(content=f"Document `{file_name}` not found in index.").send()
            return

        path = Path(match[0]["source_path"])
        await asyncio.to_thread(self._pipeline.remove, path)
        await cl.Message(content=f"Deleted from index: {path.name}").send()

    async def _handle_history(self) -> None:
        threads = self._sqlite.list_threads(limit=20)
        if not threads:
            await cl.Message(content="No conversation history.").send()
            return

        lines = ["## Conversation History\n"]
        for t in threads:
            display = t["summary"] or t["title"] or "Untitled"
            date = t["updated_at"][:16].replace("T", " ")
            lines.append(f"- **{display}** — {date} `[{t['id'][:8]}]`")
        await cl.Message(content="\n".join(lines)).send()

    async def _handle_compress(self) -> None:
        thread_id = cl.user_session.get("thread_id")
        if not thread_id:
            await cl.Message(content="No active thread to compress.").send()
            return

        messages = self._sqlite.get_thread_messages(thread_id)
        if not messages:
            await cl.Message(content="Thread is empty.").send()
            return

        conversation = "\n".join(
            f"{m['role']}: {m['content'][:300]}" for m in messages[:20]
        )
        prompt = (
            "Summarize this conversation in 1-2 concise sentences capturing the key topics discussed:\n\n"
            f"{conversation}"
        )
        summary = await asyncio.to_thread(self._engine.complete, prompt)
        if not summary:
            summary = messages[0]["content"][:50] if messages else "Empty thread"
        self._sqlite.update_thread_summary(thread_id, summary)
        await cl.Message(content="Thread compressed.").send()
