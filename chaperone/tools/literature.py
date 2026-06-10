"""External knowledge tools: arXiv and web search.

Calls the ``arxiv`` and ``ddgs`` libraries directly (no langchain-community tool
wrappers). The clients are imported lazily inside each tool so importing this module
stays cheap and free of network/validation side effects.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

_MAX_RESULTS = 5
_SUMMARY_CHARS = 400


class QueryArgs(BaseModel):
    query: str = Field(description="Free-text search query.")


@tool(args_schema=QueryArgs)
def search_literature(query: str) -> str:
    """Search arXiv for preprints (protein engineering, ML, structural biology).

    Use for specific papers or advances not in the local corpus.
    """
    import arxiv

    search = arxiv.Search(query=query, max_results=_MAX_RESULTS)
    blocks: list[str] = []
    for r in arxiv.Client().results(search):
        authors = ", ".join(a.name for a in r.authors[:3])
        year = r.published.year if r.published else "n.d."
        blocks.append(f"{r.title} ({year}) — {authors}\n{r.summary.strip()[:_SUMMARY_CHARS]}\n{r.entry_id}")
    return "\n\n".join(blocks) if blocks else "No arXiv results."


@tool(args_schema=QueryArgs)
def web_search(query: str) -> str:
    """Web search via DuckDuckGo for docs, tutorials, or tool syntax.

    Use to look up an external API or check an online biological database.
    """
    from ddgs import DDGS

    hits = DDGS().text(query, max_results=_MAX_RESULTS)
    blocks = [
        f"{h.get('title', '')}\n{h.get('body', '')}\n{h.get('href', '')}".strip() for h in hits
    ]
    return "\n\n".join(blocks) if blocks else "No web results."
