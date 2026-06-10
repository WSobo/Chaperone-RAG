"""Pydantic v2 domain models — the typed contract between every pipeline stage.

Every function that crosses a stage boundary (ingest → retrieve → rerank → answer)
takes and returns one of these models, never a bare dict, tuple, or string. The LLM's
output is parsed into :class:`Answer` so generation is data, not free text.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceType(str, Enum):
    """Where a piece of corpus content came from."""

    pdf = "pdf"
    web = "web"
    pdb = "pdb"
    manual = "manual"


class IngestSource(BaseModel):
    """A single thing to ingest: a local file/dir path or a URL."""

    model_config = ConfigDict(frozen=True)

    uri: str
    source_type: SourceType
    title: str | None = None


class Chunk(BaseModel):
    """A retrievable unit of text plus provenance."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    source_uri: str
    source_type: SourceType = SourceType.manual
    title: str | None = None
    page: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    def short_source(self) -> str:
        """Human-friendly source label for citations."""
        label = self.title or self.source_uri
        return f"{label} (p.{self.page})" if self.page is not None else label


class RetrievedChunk(BaseModel):
    """A chunk surfaced by retrieval, with its relevance score and final rank."""

    chunk: Chunk
    score: float = Field(description="Final relevance score (post-rerank if reranking is on).")
    rank: int = Field(ge=0, description="0-based rank after the full retrieval pipeline.")
    retriever: str = Field(default="hybrid", description="Stage/strategy that surfaced it.")


class Citation(BaseModel):
    """A grounded reference backing a span of the answer."""

    marker: int = Field(ge=1, description="The [n] marker used inline in the answer text.")
    source_uri: str
    title: str | None = None
    locator: str | None = Field(default=None, description='e.g. "p.4".')
    quote: str | None = Field(default=None, description="Supporting span from the source.")


class QuerySpec(BaseModel):
    """A user question expanded for retrieval (multi-query / HyDE)."""

    original: str
    sub_queries: list[str] = Field(default_factory=list)
    hyde_doc: str | None = None

    @field_validator("sub_queries")
    @classmethod
    def _always_include_original(cls, v: list[str], info) -> list[str]:  # type: ignore[no-untyped-def]
        # De-duplicate while preserving order; the original is guaranteed to be searched
        # by the retriever, so we only keep distinct non-empty expansions here.
        seen: set[str] = set()
        out: list[str] = []
        for q in v:
            q = q.strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        return out

    def all_queries(self) -> list[str]:
        """Original question first, then de-duplicated expansions."""
        out = [self.original]
        seen = {self.original.lower()}
        for q in self.sub_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        return out


class Answer(BaseModel):
    """The validated output of the RAG chain.

    ``grounded=False`` is the honest "I can't answer from the corpus" signal — the
    chain sets it when retrieved context does not support an answer, rather than
    letting the model improvise.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    def cited_markers(self) -> set[int]:
        return {c.marker for c in self.citations}
