"""Chroma-backed dense vector store, persisted to disk.

Wraps langchain-chroma so the rest of the app depends on a small typed surface,
not Chroma directly. Also exposes ``all_documents`` because the BM25 half of the
hybrid retriever needs the full corpus in memory.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from chaperone.settings import Settings
from chaperone.utils.logger import logger


class VectorStore:
    def __init__(self, settings: Settings, embeddings: Embeddings) -> None:
        self._settings = settings
        self._store = Chroma(
            collection_name=settings.retrieval.collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.paths.vector_db),
        )

    def add(self, documents: list[Document]) -> int:
        if not documents:
            return 0
        self._store.add_documents(documents)  # langchain-chroma persists automatically
        logger.info(f"Indexed {len(documents)} chunks into '{self._settings.retrieval.collection_name}'.")
        return len(documents)

    def dense_retriever(self, k: int) -> VectorStoreRetriever:
        return self._store.as_retriever(search_kwargs={"k": k})

    def all_documents(self) -> list[Document]:
        """Every stored chunk as Documents (for BM25 index construction)."""
        got = self._store.get()  # {'ids', 'documents', 'metadatas', ...}
        texts = got.get("documents") or []
        metas = got.get("metadatas") or [{}] * len(texts)
        return [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metas)]

    def count(self) -> int:
        return len(self._store.get().get("ids") or [])
