"""The RAG chain: retrieve → generate → assemble a validated, cited Answer.

Composed as an LCEL pipeline of three steps. The key invariant lives in
``_citations``: the model's claimed citation markers are validated against the
sources actually retrieved, so an answer can never cite a source that wasn't put
in front of the model. Generation is data here — the output is a Pydantic
:class:`~chaperone.schemas.Answer`, not a string.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from chaperone.llm.base import DraftAnswer, LLMBackend, parse_draft
from chaperone.rag import prompts
from chaperone.retrieval.retriever import Retriever
from chaperone.schemas import Answer, Citation, RetrievedChunk


class RAGChain:
    def __init__(self, retriever: Retriever, llm: LLMBackend, *, max_quote_chars: int = 240) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_quote_chars = max_quote_chars
        self._chain: Runnable[str, Answer] = (
            RunnableLambda(self._retrieve)
            | RunnableLambda(self._generate)
            | RunnableLambda(self._assemble)
        )

    def invoke(self, question: str) -> Answer:
        return self._chain.invoke(question)

    def answer_from_contexts(self, question: str, contexts: list[RetrievedChunk]) -> Answer:
        """Generate + assemble an Answer from already-retrieved contexts.

        Skips the retrieve step — used by the agent, which does its own (corrective)
        retrieval before delegating generation here.
        """
        state = self._generate({"question": question, "contexts": contexts})
        return self._assemble(state)

    @property
    def runnable(self) -> Runnable[str, Answer]:
        """The underlying LCEL runnable (for batching/streaming/composition)."""
        return self._chain

    # --- pipeline steps -------------------------------------------------------

    def _retrieve(self, question: str) -> dict[str, Any]:
        return {"question": question, "contexts": self._retriever.retrieve(question)}

    def _generate(self, state: dict[str, Any]) -> dict[str, Any]:
        contexts: list[RetrievedChunk] = state["contexts"]
        if not contexts:
            state["draft"] = DraftAnswer(
                text=(
                    "I couldn't find supporting passages in the corpus for that question, "
                    "so I won't answer from guesswork."
                ),
                grounded=False,
                confidence=0.0,
            )
        else:
            state["draft"] = self._draft(state["question"], contexts)
        return state

    def _assemble(self, state: dict[str, Any]) -> Answer:
        contexts: list[RetrievedChunk] = state["contexts"]
        draft: DraftAnswer = state["draft"]
        citations = self._citations(draft, contexts)
        return Answer(
            question=state["question"],
            text=draft.text,
            citations=citations,
            grounded=draft.grounded and bool(contexts),
            confidence=draft.confidence,
        )

    # --- helpers --------------------------------------------------------------

    def _draft(self, question: str, contexts: list[RetrievedChunk]) -> DraftAnswer:
        # Backends may own structuring (the mock does); otherwise prompt + parse.
        draft_fn = getattr(self._llm, "draft_answer", None)
        if callable(draft_fn):
            return draft_fn(question, contexts, system=prompts.SYSTEM)
        raw = self._llm.complete(
            prompts.build_answer_prompt(question, contexts), system=prompts.SYSTEM
        )
        return parse_draft(raw)

    def _citations(self, draft: DraftAnswer, contexts: list[RetrievedChunk]) -> list[Citation]:
        n = len(contexts)
        # Only markers that point at a source actually retrieved survive.
        markers = [m for m in draft.used_markers if 1 <= m <= n]
        if not markers and draft.grounded and contexts:
            markers = [1]  # minimum grounding: attribute to the top source
        out: list[Citation] = []
        for m in dict.fromkeys(markers):  # de-dupe, preserve order
            rc = contexts[m - 1]
            out.append(
                Citation(
                    marker=m,
                    source_uri=rc.chunk.source_uri,
                    title=rc.chunk.title,
                    locator=f"p.{rc.chunk.page}" if rc.chunk.page is not None else None,
                    quote=rc.chunk.text.strip()[: self._max_quote_chars],
                )
            )
        return out
