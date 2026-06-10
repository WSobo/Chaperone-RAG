"""Composition root — assemble the pipeline from settings.

The CLI, tests, and eval harness build their objects here so wiring lives in one
place. Ingestion deliberately avoids constructing the LLM and reranker so
``chaperone ingest`` stays light and offline.
"""

from __future__ import annotations

from chaperone.agent import ChaperoneAgent
from chaperone.ingestion import Ingestor
from chaperone.llm import get_llm
from chaperone.llm.base import LLMBackend
from chaperone.rag import RAGChain
from chaperone.retrieval import Retriever, VectorStore, build_embeddings, build_reranker
from chaperone.settings import Settings, get_settings


def build_vector_store(settings: Settings) -> VectorStore:
    return VectorStore(settings, build_embeddings(settings))


def build_ingestor(settings: Settings | None = None) -> Ingestor:
    settings = settings or get_settings()
    settings.ensure_dirs()
    return Ingestor(settings, build_vector_store(settings))


def build_retriever(
    settings: Settings,
    store: VectorStore | None = None,
    llm: LLMBackend | None = None,
) -> Retriever:
    store = store or build_vector_store(settings)
    return Retriever(settings, store, llm=llm, reranker=build_reranker(settings))


def build_rag_chain(settings: Settings | None = None) -> RAGChain:
    settings = settings or get_settings()
    settings.ensure_dirs()
    store = build_vector_store(settings)
    llm = get_llm(settings)
    retriever = build_retriever(settings, store=store, llm=llm)
    return RAGChain(retriever, llm)


def build_agent(settings: Settings | None = None) -> ChaperoneAgent:
    settings = settings or get_settings()
    settings.ensure_dirs()
    store = build_vector_store(settings)
    llm = get_llm(settings)
    retriever = build_retriever(settings, store=store, llm=llm)
    return ChaperoneAgent(settings, retriever, RAGChain(retriever, llm))
