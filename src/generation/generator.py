from collections.abc import AsyncIterator
from typing import Any

from src.retrieval.base import RetrievalResult


class Generator:
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

        messages = [{"role": "system", "content": self._system_prompt}]

        if chat_history:
            messages.extend(chat_history)

        user_content = f"""Context from documents:

{context}

---

Question: {query}"""

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
                yield chunk.content
