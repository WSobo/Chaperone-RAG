"""Prompts and context rendering for the RAG chain."""

from __future__ import annotations

from chaperone.schemas import RetrievedChunk

SYSTEM = (
    "You are Chaperone, an expert assistant for protein design and engineering "
    "(RFdiffusion, ProteinMPNN, AlphaFold/ESMFold, Rosetta). Answer ONLY from the "
    "numbered sources provided. Cite every claim with its source number like [1]. "
    "If the sources do not contain the answer, say so plainly and set grounded=false. "
    "Never use outside knowledge or invent citations."
)

_ANSWER_INSTRUCTION = (
    "Answer the question using only the numbered sources. Respond with a JSON object:\n"
    '  "text":         the answer, with inline [n] citations\n'
    '  "used_markers": list of the source numbers you cited\n'
    '  "grounded":     true if the sources support an answer, else false\n'
    '  "confidence":   your confidence from 0 to 1\n\n'
    "Question: {question}\n\nSources:\n{context}\n\nJSON:"
)


def render_context(contexts: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as the numbered source blocks the answer cites."""
    return "\n\n".join(
        f"[{i + 1}] {rc.chunk.short_source()}\n{rc.chunk.text}" for i, rc in enumerate(contexts)
    )


def build_answer_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    return _ANSWER_INSTRUCTION.format(question=question, context=render_context(contexts))
