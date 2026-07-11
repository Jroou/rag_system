from pathlib import Path


def load_pdf(path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def extract_pdf_title(path: Path) -> str | None:
    """Return the document title from PDF metadata, or the first non-empty line of page 1."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    # Try metadata first
    meta = reader.metadata
    if meta:
        title = getattr(meta, "title", None) or meta.get("/Title")
        if title and title.strip():
            return title.strip()
    # Fall back to first meaningful line of page 1
    if reader.pages:
        text = reader.pages[0].extract_text() or ""
        for line in text.splitlines():
            line = line.strip()
            if len(line) > 10:
                return line
    return None


def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_code(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_docx(path: Path) -> list[str]:
    from docx import Document

    doc = Document(str(path))
    return [para.text for para in doc.paragraphs]


def detect_document_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    elif suffix == ".md":
        return "markdown"
    elif suffix in (".doc", ".docx"):
        return "docx"
    elif suffix in (".py", ".js", ".ts", ".java", ".go", ".rs", ".rb", ".cpp", ".c"):
        return "code"
    else:
        return "unknown"
