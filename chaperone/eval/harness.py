"""Evaluation harness.

Two layers:
  1. Deterministic *retrieval* metrics (hit-rate, MRR, grounded-rate) computed against
     the labeled golden set. These always run — no judge LLM, no network beyond the
     embedding model — so they're safe for CI and catch retrieval regressions.
  2. Optional RAGAS *generation* metrics (faithfulness, answer relevancy, context
     precision/recall). These need an evaluator LLM; if RAGAS or its judge isn't
     available the harness reports the retrieval metrics alone and says so.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel

from chaperone.eval.golden_set import CORPUS, GOLDEN
from chaperone.llm import get_llm
from chaperone.rag import RAGChain
from chaperone.retrieval import Retriever, VectorStore, build_embeddings, build_reranker
from chaperone.settings import Settings, get_settings
from chaperone.utils.logger import logger


class ItemResult(BaseModel):
    question: str
    hit: bool
    reciprocal_rank: float
    grounded: bool
    n_citations: int


class EvalReport(BaseModel):
    items: list[ItemResult]
    hit_rate: float
    mrr: float
    grounded_rate: float
    ragas: dict[str, float] | None = None

    def summary(self) -> str:
        lines = [
            f"hit_rate      {self.hit_rate:.2f}",
            f"mrr           {self.mrr:.2f}",
            f"grounded_rate {self.grounded_rate:.2f}",
        ]
        if self.ragas:
            lines += [f"{k:<13} {v:.2f}" for k, v in self.ragas.items()]
        else:
            lines.append("ragas         (skipped — needs an evaluator LLM)")
        return "\n".join(lines)


def build_eval_pipeline(settings: Settings) -> tuple[Retriever, RAGChain]:
    """Build an ephemeral retriever + RAG chain over the bundled golden corpus."""
    tmp = tempfile.mkdtemp(prefix="chaperone_eval_")
    eval_settings = settings.model_copy(
        update={
            "paths": settings.paths.model_copy(update={"vector_db": Path(tmp)}),
            "retrieval": settings.retrieval.model_copy(update={"collection_name": "chaperone_eval"}),
        }
    )
    store = VectorStore(eval_settings, build_embeddings(eval_settings))
    store.add(
        [
            Document(
                page_content=text,
                metadata={"source": doc_id, "chunk_id": doc_id, "title": doc_id, "source_type": "manual"},
            )
            for doc_id, text in CORPUS
        ]
    )
    llm = get_llm(eval_settings)
    retriever = Retriever(eval_settings, store, llm=llm, reranker=build_reranker(eval_settings))
    return retriever, RAGChain(retriever, llm)


def run_eval(
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    rag: RAGChain | None = None,
) -> EvalReport:
    settings = settings or get_settings()
    if retriever is None or rag is None:
        retriever, rag = build_eval_pipeline(settings)

    items: list[ItemResult] = []
    ragas_rows: list[dict] = []
    for g in GOLDEN:
        retrieved = retriever.retrieve(g.question)
        ranked_ids = [rc.chunk.source_uri for rc in retrieved]
        rr = _reciprocal_rank(ranked_ids, g.relevant_doc_ids)
        answer = rag.answer_from_contexts(g.question, retrieved)
        items.append(
            ItemResult(
                question=g.question,
                hit=rr > 0,
                reciprocal_rank=rr,
                grounded=answer.grounded,
                n_citations=len(answer.citations),
            )
        )
        ragas_rows.append(
            {
                "question": g.question,
                "answer": answer.text,
                "contexts": [rc.chunk.text for rc in retrieved],
                "ground_truth": g.ideal_answer,
            }
        )

    n = len(items) or 1
    return EvalReport(
        items=items,
        hit_rate=sum(i.hit for i in items) / n,
        mrr=sum(i.reciprocal_rank for i in items) / n,
        grounded_rate=sum(i.grounded for i in items) / n,
        ragas=_try_ragas(ragas_rows),
    )


def _reciprocal_rank(ranked_ids: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def _try_ragas(rows: list[dict]) -> dict[str, float] | None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        logger.info("RAGAS not installed; reporting retrieval metrics only (pip install -e '.[eval]').")
        return None

    try:
        result = evaluate(
            Dataset.from_list(rows),
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        scores = result.to_pandas().mean(numeric_only=True).to_dict()
        return {k: float(v) for k, v in scores.items()}
    except Exception as e:  # missing judge LLM / API key, etc.
        logger.warning(f"RAGAS skipped (needs an evaluator LLM/API key): {e}")
        return None
