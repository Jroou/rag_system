from pathlib import Path


def load_pdf(path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


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
