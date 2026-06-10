"""Deterministic, CPU-only LLM backend.

This is what makes Chaperone-RAG demoable and testable without a GPU. Instead of
hallucinating, :class:`MockLLM` does *extractive* synthesis over the retrieved
context: it picks the sentence in each top source that best matches the question
and stitches them into a grounded, cited answer. No weights, no network, fully
deterministic — so the retrieval pipeline and the citation contract can be
exercised end-to-end in CI.
"""

from __future__ import annotations

import re

from chaperone.llm.base import DraftAnswer
from chaperone.schemas import RetrievedChunk

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in into is it its of on or that
    the to was were what when where which who why will with you your we our can does do
    this these those using use used based""".split()
)


def _keywords(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in _STOPWORDS and len(w) > 2}


def _best_sentence(text: str, keywords: set[str]) -> tuple[str | None, int]:
    """Return the highest-overlap sentence and its overlap score (0 if none match)."""
    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    if not sentences:
        return None, 0
    best, best_score = sentences[0], 0
    for s in sentences:
        score = len(_keywords(s) & keywords)
        if score > best_score:
            best, best_score = s, score
    return best, best_score


class MockLLM:
    """Extractive, deterministic stand-in for a real generative model."""

    name = "mock"

    def __init__(self, max_sources: int = 3, max_quote_chars: int = 320) -> None:
        self.max_sources = max_sources
        self.max_quote_chars = max_quote_chars

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        # Generic completions (e.g. from tools) are not the mock's strength; return a
        # transparent, deterministic echo. The RAG chain uses ``draft_answer`` instead.
        head = " ".join(prompt.split())[:240]
        return f"[mock-llm] {head}"

    def draft_answer(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        *,
        system: str | None = None,
    ) -> DraftAnswer:
        if not contexts:
            return DraftAnswer(
                text=(
                    "I couldn't find supporting passages in the corpus, so I won't answer "
                    "this from guesswork. Try ingesting a relevant paper or rephrasing."
                ),
                used_markers=[],
                grounded=False,
                confidence=0.0,
            )

        keywords = _keywords(question)
        parts: list[str] = []
        markers: list[int] = []
        for i, rc in enumerate(contexts[: self.max_sources]):
            sentence, score = _best_sentence(rc.chunk.text, keywords)
            if not sentence or score == 0:
                continue  # only cite sources that actually match the question
            quote = sentence[: self.max_quote_chars].rstrip()
            parts.append(f"{quote} [{i + 1}]")
            markers.append(i + 1)

        if not parts:  # nothing matched: ground in the single top-ranked source
            lead = contexts[0].chunk.text.strip()[: self.max_quote_chars]
            parts = [f"{lead} [1]"]
            markers = [1]

        text = "Based on the retrieved sources: " + " ".join(parts)
        # Confidence scales with how many sources corroborate and the top score.
        top_score = max(0.0, min(1.0, contexts[0].score))
        confidence = round(min(0.9, 0.35 + 0.12 * len(markers) + 0.2 * top_score), 2)
        return DraftAnswer(text=text, used_markers=markers, grounded=True, confidence=confidence)
