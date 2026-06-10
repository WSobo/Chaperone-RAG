"""Hybrid retrieval: dense vectors fused with BM25 lexical search via RRF.

Protein-design queries are full of exact tokens that dense embeddings blur together —
gene names, PDB IDs, "ProteinMPNN", "ipTM", "pLDDT". BM25 nails those; dense search
handles paraphrase and concept overlap. We fuse the two ranked lists with weighted
**reciprocal-rank fusion**.

Both halves are first-party here: BM25 is a thin wrapper over ``rank_bm25`` and the
fusion is explicit RRF. No langchain-community retriever in the path — fewer deps, and
the ranking is fully inspectable.
"""

from __future__ import annotations

import hashlib
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from chaperone.retrieval.vectorstore import VectorStore
from chaperone.settings import Settings

_RRF_C = 60  # standard RRF damping constant: score contribution = weight / (C + rank)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25Index:
    """Lexical search over a fixed corpus via Okapi BM25 (rank_bm25)."""

    def __init__(self, documents: list[Document], k: int) -> None:
        self._docs = documents
        self._k = k
        self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in documents])

    def search(self, query: str) -> list[Document]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(self._docs)), key=lambda i: scores[i], reverse=True)
        return [self._docs[i] for i in order[: self._k]]


class HybridRetriever:
    """Dense + BM25 retrieval fused by weighted reciprocal-rank fusion."""

    def __init__(
        self,
        dense,  # langchain VectorStoreRetriever (duck-typed: has .invoke)
        bm25: BM25Index | None,
        k: int,
        dense_weight: float,
        bm25_weight: float,
    ) -> None:
        self._dense = dense
        self._bm25 = bm25
        self._k = k
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight

    def invoke(self, query: str) -> list[Document]:
        dense_docs = self._dense.invoke(query)
        if self._bm25 is None:  # empty corpus / BM25 unavailable → dense only
            return dense_docs[: self._k]
        bm25_docs = self._bm25.search(query)
        return self._fuse([(dense_docs, self._dense_weight), (bm25_docs, self._bm25_weight)])

    def _fuse(self, ranked_lists: list[tuple[list[Document], float]]) -> list[Document]:
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for ranked, weight in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = _doc_key(doc)
                scores[key] = scores.get(key, 0.0) + weight / (_RRF_C + rank + 1)
                docs.setdefault(key, doc)
        ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
        return [docs[key] for key in ordered[: self._k]]


def build_hybrid_retriever(vector_store: VectorStore, settings: Settings) -> HybridRetriever:
    k = settings.retrieval.top_k
    dense = vector_store.dense_retriever(k)

    # BM25 needs the corpus materialized; on an empty store fall back to dense only.
    corpus = vector_store.all_documents()
    bm25 = BM25Index(corpus, k) if corpus else None

    return HybridRetriever(
        dense, bm25, k, settings.retrieval.dense_weight, settings.retrieval.bm25_weight
    )


def _doc_key(doc: Document) -> str:
    md = doc.metadata or {}
    return str(md.get("chunk_id") or hashlib.sha1(doc.page_content.encode()).hexdigest()[:16])
