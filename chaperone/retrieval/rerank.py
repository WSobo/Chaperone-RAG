"""Cross-encoder reranking — the single biggest precision lever.

Bi-encoder retrieval (dense/BM25) scores query and document independently; a
cross-encoder reads the (query, passage) pair jointly and is far better at telling
"mentions RFdiffusion" from "actually explains RFdiffusion conditioning". We
over-retrieve with the cheap hybrid stage, then rerank down to ``rerank_top_n``.
"""

from __future__ import annotations

from langchain_core.documents import Document

from chaperone.settings import Settings


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder  # lazy: heavy import

        self._model = CrossEncoder(model_name, device=device)

    def rerank(
        self, query: str, documents: list[Document], top_n: int
    ) -> list[tuple[Document, float]]:
        if not documents:
            return []
        scores = self._model.predict([(query, d.page_content) for d in documents])
        order = sorted(range(len(documents)), key=lambda i: float(scores[i]), reverse=True)
        return [(documents[i], float(scores[i])) for i in order[:top_n]]


def build_reranker(settings: Settings) -> CrossEncoderReranker | None:
    if not settings.retrieval.use_reranker:
        return None
    return CrossEncoderReranker(settings.retrieval.reranker_model, device=settings.embedding.device)
