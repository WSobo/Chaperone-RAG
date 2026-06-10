"""Source loaders: turn paths/URLs into LangChain Documents with clean provenance.

Implemented directly on the underlying libraries (``pypdf`` for PDFs, ``requests`` +
``BeautifulSoup`` for web) rather than via langchain-community loaders — fewer moving
parts, a smaller dependency tree, and full control over what text and metadata we keep.
We still emit ``langchain_core.documents.Document`` because that's what the splitters,
Chroma, and BM25 fusion downstream all speak.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader

from chaperone.schemas import IngestSource, SourceType
from chaperone.utils.logger import logger

_TEXT_SUFFIXES = {".txt", ".md", ".rst"}
_HTTP_TIMEOUT = 30


def source_from_str(raw: str) -> IngestSource:
    """Classify a raw CLI argument as a URL, a PDF/dir, or a text file."""
    if raw.startswith(("http://", "https://")):
        return IngestSource(uri=raw, source_type=SourceType.web)
    p = Path(raw)
    if p.is_dir() or p.suffix.lower() == ".pdf":
        return IngestSource(uri=str(p), source_type=SourceType.pdf)
    return IngestSource(uri=str(p), source_type=SourceType.manual)


def load_source(source: IngestSource) -> list[Document]:
    if source.source_type == SourceType.web:
        docs = _load_url(source.uri)
    else:
        path = Path(source.uri)
        if path.is_dir():
            docs = _load_pdf_dir(path)
        elif path.suffix.lower() == ".pdf":
            docs = _load_pdf(path)
        elif path.suffix.lower() in _TEXT_SUFFIXES:
            docs = _load_text(path)
        else:
            logger.warning(f"Unsupported source skipped: {source.uri}")
            return []

    for d in docs:
        _finalize(d, source)
    logger.info(f"Loaded {len(docs)} document(s) from {source.uri}")
    return docs


def load_sources(sources: list[IngestSource]) -> list[Document]:
    out: list[Document] = []
    for s in sources:
        out.extend(load_source(s))
    return out


# --- per-type loaders ---------------------------------------------------------


def _load_pdf(path: Path) -> list[Document]:
    """One Document per page (matching the page-level granularity citations expect)."""
    reader = PdfReader(str(path))
    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(
                Document(page_content=text, metadata={"source": str(path), "page": i, "title": path.name})
            )
    return docs


def _load_pdf_dir(path: Path) -> list[Document]:
    docs: list[Document] = []
    for pdf in sorted(path.glob("*.pdf")):
        docs.extend(_load_pdf(pdf))
    return docs


def _load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": str(path), "title": path.name})]


def _load_url(url: str) -> list[Document]:
    headers = {"User-Agent": os.environ.get("USER_AGENT", "Chaperone-RAG")}
    resp = requests.get(url, timeout=_HTTP_TIMEOUT, headers=headers)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    if not text:
        return []

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    return [Document(page_content=text, metadata={"source": url, "title": title})]


# --- provenance ---------------------------------------------------------------


def _finalize(doc: Document, source: IngestSource) -> None:
    md = doc.metadata or {}
    md.setdefault("source", source.uri)
    md["source_type"] = source.source_type.value
    md.setdefault("title", source.title or _default_title(str(md["source"])))
    doc.metadata = md


def _default_title(source: str) -> str:
    if source.startswith(("http://", "https://")):
        return source
    return Path(source).name
