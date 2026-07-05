from dataclasses import dataclass, field

from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


@dataclass
class Chunk:
    text: str
    chunk_type: str  # "parent" or "child"
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)


def _split_into_parent_child(
    texts: list[str],
    parent_chunk_size: int,
    child_chunk_size: int,
    chunk_overlap: int,
    base_metadata: dict,
) -> list[Chunk]:
    import uuid

    chunks = []

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    full_text = "\n\n".join(texts)
    parent_docs = parent_splitter.split_text(full_text)

    for parent_text in parent_docs:
        parent_id = str(uuid.uuid4())
        chunks.append(
            Chunk(
                text=parent_text,
                chunk_type="parent",
                parent_id=None,
                metadata={**base_metadata, "chunk_id": parent_id},
            )
        )

        child_texts = child_splitter.split_text(parent_text)
        for child_text in child_texts:
            child_id = str(uuid.uuid4())
            chunks.append(
                Chunk(
                    text=child_text,
                    chunk_type="child",
                    parent_id=parent_id,
                    metadata={**base_metadata, "chunk_id": child_id},
                )
            )

    return chunks


def chunk_markdown(
    text: str,
    parent_chunk_size: int = 1000,
    child_chunk_size: int = 200,
    chunk_overlap: int = 50,
    metadata: dict | None = None,
) -> list[Chunk]:
    base_metadata = metadata or {}
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    sections = splitter.split_text(text)
    section_texts = [doc.page_content for doc in sections]

    if not section_texts:
        section_texts = [text]

    return _split_into_parent_child(
        section_texts, parent_chunk_size, child_chunk_size, chunk_overlap, base_metadata
    )


def chunk_code(
    text: str,
    language: Language,
    parent_chunk_size: int = 1000,
    child_chunk_size: int = 200,
    chunk_overlap: int = 50,
    metadata: dict | None = None,
) -> list[Chunk]:
    base_metadata = metadata or {}
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=parent_chunk_size,
        chunk_overlap=chunk_overlap,
    )
    sections = splitter.split_text(text)

    if not sections:
        sections = [text]

    return _split_into_parent_child(
        sections, parent_chunk_size, child_chunk_size, chunk_overlap, base_metadata
    )


def chunk_pdf(
    pages: list[str],
    parent_chunk_size: int = 1000,
    child_chunk_size: int = 200,
    chunk_overlap: int = 50,
    metadata: dict | None = None,
) -> list[Chunk]:
    base_metadata = metadata or {}
    return _split_into_parent_child(
        pages, parent_chunk_size, child_chunk_size, chunk_overlap, base_metadata
    )


def chunk_docx(
    paragraphs: list[str],
    parent_chunk_size: int = 1000,
    child_chunk_size: int = 200,
    chunk_overlap: int = 50,
    metadata: dict | None = None,
) -> list[Chunk]:
    base_metadata = metadata or {}
    non_empty = [p for p in paragraphs if p.strip()]
    if not non_empty:
        non_empty = [""]
    return _split_into_parent_child(
        non_empty, parent_chunk_size, child_chunk_size, chunk_overlap, base_metadata
    )


EXTENSION_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".rb": Language.RUBY,
    ".cpp": Language.CPP,
    ".c": Language.C,
}
