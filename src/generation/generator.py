from collections.abc import AsyncIterator
from typing import Any

from src.retrieval.base import RetrievalResult


class Generator:
    CITATION_INSTRUCTION = (
        "\n\nWhen answering, cite your sources using inline numbered references like [1], [2], etc. "
        "Each number corresponds to the source number shown in the context. "
        "Only cite sources you actually use. "
        "Every claim in your response must be directly supported by a cited source — do not extrapolate or infer beyond what the sources state."
    )

    def __init__(self, llm, system_prompt: str):
        self._llm = llm
        self._system_prompt = system_prompt

    def build_prompt(
        self, query: str, results: list[RetrievalResult], chat_history: list[dict] | None = None
    ) -> list[dict]:
        context_parts = []
        for i, r in enumerate(results, 1):
            text = r.parent_text or r.text
            context_parts.append(
                f"[Source {i}: {r.source_path}]\n{text}"
            )
        context = "\n\n---\n\n".join(context_parts)

        system_content = self._system_prompt
        if results:
            system_content += self.CITATION_INSTRUCTION

        messages = [{"role": "system", "content": system_content}]

        if chat_history:
            messages.extend(chat_history)

        user_content = f"""Context from {len(results)} sources:

{context}

---

Question: {query}

Use information from all {len(results)} sources above where relevant."""

        messages.append({"role": "user", "content": user_content})
        return messages

    def generate(
        self, query: str, results: list[RetrievalResult], chat_history: list[dict] | None = None
    ) -> str:
        messages = self.build_prompt(query, results, chat_history)
        response = self._llm.invoke(messages)
        return response.content

    async def agenerate_stream(
        self, query: str, results: list[RetrievalResult], chat_history: list[dict] | None = None
    ) -> AsyncIterator[str]:
        messages = self.build_prompt(query, results, chat_history)
        async for chunk in self._llm.astream(messages):
            if chunk.content:
                text = chunk.content if isinstance(chunk.content, str) else "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in chunk.content
                )
                if text:
                    yield text
