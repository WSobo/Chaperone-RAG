"""End-to-end RAG pipeline test — offline and deterministic.

Uses DeterministicFakeEmbedding (no model download / network) and the MockLLM, so
it exercises VectorStore → hybrid → retriever → RAGChain → cited Answer in CI.
Skipped automatically if the vector-store deps aren't installed.
"""

import pytest

pytest.importorskip("langchain_chroma")
pytest.importorskip("chromadb")
pytest.importorskip("rank_bm25")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings.fake import DeterministicFakeEmbedding  # noqa: E402

from chaperone.llm.mock import MockLLM  # noqa: E402
from chaperone.rag import RAGChain  # noqa: E402
from chaperone.retrieval import Retriever, VectorStore  # noqa: E402
from chaperone.settings import Settings  # noqa: E402


def _settings(tmp_path):
    s = Settings()
    s.paths.vector_db = tmp_path / "db"
    s.retrieval.collection_name = "test_docs"
    s.retrieval.use_reranker = False  # no cross-encoder download in tests
    s.retrieval.use_multi_query = False
    s.retrieval.use_hyde = False
    return s


def _store(settings):
    return VectorStore(settings, DeterministicFakeEmbedding(size=64))


def _doc(doc_id, text, title):
    return Document(page_content=text, metadata={"source": doc_id, "chunk_id": doc_id, "title": title})


def test_pipeline_returns_grounded_cited_answer(tmp_path):
    s = _settings(tmp_path)
    store = _store(s)
    store.add(
        [
            _doc("rfdiff", "RFdiffusion is a diffusion model for protein backbone and binder design.", "RFdiffusion"),
            _doc("mpnn", "ProteinMPNN designs amino acid sequences for a fixed backbone.", "ProteinMPNN"),
        ]
    )
    rag = RAGChain(Retriever(s, store, llm=MockLLM(), reranker=None), MockLLM())

    answer = rag.invoke("What is RFdiffusion used for in protein backbone design?")
    assert answer.grounded
    assert answer.citations, "a grounded answer must carry at least one citation"
    # Every citation must point at a source that was actually retrieved.
    for c in answer.citations:
        assert c.marker >= 1
        assert c.source_uri in {"rfdiff", "mpnn"}


def test_empty_corpus_is_ungrounded(tmp_path):
    s = _settings(tmp_path)
    rag = RAGChain(Retriever(s, _store(s), llm=MockLLM(), reranker=None), MockLLM())
    answer = rag.invoke("a question with no corpus to answer from")
    assert answer.grounded is False
    assert answer.citations == []
