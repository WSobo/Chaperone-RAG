"""Document chunking with stable, content-addressed chunk ids.

The id is a hash of (source, page, text) so re-ingesting the same content is
idempotent at the chunk level and citations stay stable across runs.
"""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chaperone.settings import Settings


def make_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.retrieval.chunk_size,
        chunk_overlap=settings.retrieval.chunk_overlap,
        add_start_index=True,
    )


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    chunks = make_splitter(settings).split_documents(documents)
    for c in chunks:
        md = c.metadata or {}
        md["chunk_id"] = _chunk_id(str(md.get("source", "unknown")), md.get("page"), c.page_content)
        c.metadata = md
    return chunks


def _chunk_id(source: str, page: object, text: str) -> str:
    h = hashlib.sha1(f"{source}|{page}|{text}".encode()).hexdigest()
    return h[:16]
