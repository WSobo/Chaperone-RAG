"""Composition root — assemble the pipeline from settings.

The CLI, tests, and eval harness build their objects here so wiring lives in one
place. Heavy imports (the retrieval/LLM stack) are deferred into each builder so that
importing this module — and running the lightweight job/skills commands — does not
pull in torch/langchain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chaperone.settings import Settings, get_settings

if TYPE_CHECKING:  # type-only imports; never executed at runtime
    from chaperone.agent import ChaperoneAgent
    from chaperone.ingestion import Ingestor
    from chaperone.jobs import JobRunner
    from chaperone.llm.base import LLMBackend
    from chaperone.rag import RAGChain
    from chaperone.retrieval import Retriever, VectorStore
    from chaperone.skills import SkillRegistry


def build_vector_store(settings: Settings) -> VectorStore:
    from chaperone.retrieval import VectorStore, build_embeddings

    return VectorStore(settings, build_embeddings(settings))


def build_ingestor(settings: Settings | None = None) -> Ingestor:
    from chaperone.ingestion import Ingestor

    settings = settings or get_settings()
    settings.ensure_dirs()
    return Ingestor(settings, build_vector_store(settings))


def build_retriever(
    settings: Settings,
    store: VectorStore | None = None,
    llm: LLMBackend | None = None,
) -> Retriever:
    from chaperone.retrieval import Retriever, build_reranker

    store = store or build_vector_store(settings)
    return Retriever(settings, store, llm=llm, reranker=build_reranker(settings))


def build_rag_chain(settings: Settings | None = None) -> RAGChain:
    from chaperone.llm import get_llm
    from chaperone.rag import RAGChain

    settings = settings or get_settings()
    settings.ensure_dirs()
    store = build_vector_store(settings)
    llm = get_llm(settings)
    retriever = build_retriever(settings, store=store, llm=llm)
    return RAGChain(retriever, llm)


def build_agent(settings: Settings | None = None) -> ChaperoneAgent:
    from chaperone.agent import ChaperoneAgent
    from chaperone.llm import get_llm
    from chaperone.rag import RAGChain

    settings = settings or get_settings()
    settings.ensure_dirs()
    store = build_vector_store(settings)
    llm = get_llm(settings)
    retriever = build_retriever(settings, store=store, llm=llm)
    return ChaperoneAgent(settings, retriever, RAGChain(retriever, llm))


def build_registry(settings: Settings | None = None) -> SkillRegistry:
    from chaperone.skills import SkillRegistry

    settings = settings or get_settings()
    return SkillRegistry(settings.paths.skills_dir)


def build_job_runner(settings: Settings | None = None) -> JobRunner:
    from chaperone.jobs import JobRunner, LocalScheduler, SlurmScheduler

    settings = settings or get_settings()
    settings.ensure_dirs()
    scheduler = SlurmScheduler() if settings.jobs.scheduler == "slurm" else LocalScheduler()
    return JobRunner(settings, scheduler)
