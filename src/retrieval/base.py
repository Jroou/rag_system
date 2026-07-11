from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    source_path: str
    document_type: str
    parent_text: str | None = None
    document_id: str | None = None


class BaseStrategy(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 20, thread_id: str | None = None) -> list[RetrievalResult]:
        ...
