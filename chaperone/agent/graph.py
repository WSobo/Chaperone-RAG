"""Corrective-RAG agent (LangGraph).

A small StateGraph that adds a self-correction loop on top of the plain RAG chain:

    retrieve → grade → (relevant?) → generate
                          └ (weak)  → web_fallback → generate

When local retrieval looks weak, it pulls external context via the web_search tool
before answering, instead of confidently answering from poor context. The plain
``RAGChain`` remains the default fast path; reach for the agent when a single
retrieval pass may not be enough.

Relevance grading here is a deterministic keyword heuristic so the whole graph runs
under the CPU mock backend. With a tool-calling-capable backend you'd swap the
heuristic router for model-driven tool selection.
"""

from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from chaperone.rag.chain import RAGChain
from chaperone.retrieval.retriever import Retriever
from chaperone.schemas import Answer, Chunk, RetrievedChunk, SourceType
from chaperone.settings import Settings
from chaperone.tools.literature import web_search
from chaperone.utils.logger import logger

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")


class AgentState(TypedDict, total=False):
    question: str
    contexts: list[RetrievedChunk]
    relevant: bool
    used_web: bool
    answer: Answer


class ChaperoneAgent:
    def __init__(self, settings: Settings, retriever: Retriever, rag: RAGChain) -> None:
        self._settings = settings
        self._retriever = retriever
        self._rag = rag
        self._graph = self._build()

    def invoke(self, question: str) -> Answer:
        result = self._graph.invoke({"question": question})
        return result["answer"]

    # --- graph wiring ---------------------------------------------------------

    def _build(self):  # noqa: ANN202 - langgraph CompiledGraph
        g = StateGraph(AgentState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("web_fallback", self._web_fallback)
        g.add_node("generate", self._generate)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges(
            "grade", self._route, {"generate": "generate", "web_fallback": "web_fallback"}
        )
        g.add_edge("web_fallback", "generate")
        g.add_edge("generate", END)
        return g.compile()

    # --- nodes ----------------------------------------------------------------

    def _retrieve(self, state: AgentState) -> AgentState:
        return {"contexts": self._retriever.retrieve(state["question"])}

    def _grade(self, state: AgentState) -> AgentState:
        return {"relevant": _looks_relevant(state["question"], state.get("contexts", []))}

    def _route(self, state: AgentState) -> str:
        return "generate" if state.get("relevant") else "web_fallback"

    def _web_fallback(self, state: AgentState) -> AgentState:
        logger.info("Local context weak; falling back to web search.")
        contexts = list(state.get("contexts", []))
        try:
            text = web_search.invoke({"query": state["question"]})
        except Exception as e:  # network/tool failure shouldn't crash the graph
            logger.warning(f"web_search failed: {e}")
            return {"used_web": False}
        if text:
            chunk = Chunk(
                id="web-fallback",
                text=str(text),
                source_uri="duckduckgo:web_search",
                source_type=SourceType.web,
                title="Web search",
            )
            contexts.append(RetrievedChunk(chunk=chunk, score=0.3, rank=len(contexts), retriever="web"))
        return {"contexts": contexts, "used_web": True}

    def _generate(self, state: AgentState) -> AgentState:
        answer = self._rag.answer_from_contexts(state["question"], state.get("contexts", []))
        return {"answer": answer}


def _looks_relevant(question: str, contexts: list[RetrievedChunk], min_overlap: int = 2) -> bool:
    if not contexts:
        return False
    q = {w.lower() for w in _WORD.findall(question) if len(w) > 2}
    for rc in contexts:
        words = {w.lower() for w in _WORD.findall(rc.chunk.text)}
        if len(q & words) >= min_overlap:
            return True
    return False
