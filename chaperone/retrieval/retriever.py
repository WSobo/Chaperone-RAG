"""The composed retriever: transform → hybrid → rerank → typed RetrievedChunks.

This is the one object the RAG chain and the agent depend on. Everything below it
(ensemble, BM25, cross-encoder, query expansion) is an implementation detail.
"""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document

from chaperone.llm.base import LLMBackend
from chaperone.retrieval.hybrid import build_hybrid_retriever
from chaperone.retrieval.query_transform import transform_query
from chaperone.retrieval.rerank import CrossEncoderReranker
from chaperone.retrieval.vectorstore import VectorStore
from chaperone.schemas import Chunk, QuerySpec, RetrievedChunk, SourceType
from chaperone.settings import Settings


class Retriever:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        llm: LLMBackend | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._settings = settings
        self._store = vector_store
        self._llm = llm
        self._reranker = reranker

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        spec = self._plan(question)

        # Gather a candidate pool across all query variants (over-retrieve, then rerank).
        # The hybrid retriever is rebuilt per call so BM25 reflects the current corpus
        # (the dense half already reads live from Chroma).
        hybrid = build_hybrid_retriever(self._store, self._settings)
        pool: dict[str, Document] = {}
        for q in spec.all_queries() + ([spec.hyde_doc] if spec.hyde_doc else []):
            for doc in hybrid.invoke(q):
                pool.setdefault(_doc_key(doc), doc)

        candidates = list(pool.values())
        if not candidates:
            return []

        top_n = self._settings.retrieval.rerank_top_n
        if self._reranker and self._settings.retrieval.use_reranker:
            ranked = self._reranker.rerank(question, candidates, top_n)
            strategy = "hybrid+rerank"
        else:
            # No reranker: keep hybrid order, synthesize a descending score.
            ranked = [(d, 1.0 / (i + 1)) for i, d in enumerate(candidates[:top_n])]
            strategy = "hybrid"

        return [
            RetrievedChunk(chunk=chunk_from_document(doc), score=score, rank=i, retriever=strategy)
            for i, (doc, score) in enumerate(ranked)
        ]

    def _plan(self, question: str) -> QuerySpec:
        if self._llm is not None and (
            self._settings.retrieval.use_multi_query or self._settings.retrieval.use_hyde
        ):
            return transform_query(question, self._llm, self._settings)
        return QuerySpec(original=question)


def chunk_from_document(doc: Document) -> Chunk:
    md = doc.metadata or {}
    source = str(md.get("source") or md.get("source_uri") or "unknown")
    return Chunk(
        id=str(md.get("chunk_id") or _hash(doc.page_content)),
        text=doc.page_content,
        source_uri=source,
        source_type=_source_type(md.get("source_type")),
        title=md.get("title"),
        page=_as_page(md.get("page")),
        metadata={"source": source},
    )


def _source_type(value: object) -> SourceType:
    try:
        return SourceType(value) if value else SourceType.manual
    except ValueError:
        return SourceType.manual


def _as_page(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _doc_key(doc: Document) -> str:
    return str((doc.metadata or {}).get("chunk_id") or _hash(doc.page_content))


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]
