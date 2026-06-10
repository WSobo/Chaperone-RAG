"""Query transformation: multi-query expansion and HyDE.

Both fight the lexical/semantic gap between how a user phrases a question and how
papers phrase the answer. Multi-query paraphrases the question; HyDE drafts a
hypothetical answer passage and retrieves against *that*. Both degrade gracefully:
if the backend can't produce useful expansions (e.g. the CPU mock), we fall back
to searching the original question only.
"""

from __future__ import annotations

import re

from chaperone.llm.base import LLMBackend
from chaperone.schemas import QuerySpec
from chaperone.settings import Settings

_MULTIQUERY_PROMPT = (
    "You are helping retrieve protein-design literature. Rewrite the question below "
    "as 3 alternative search queries that use different terminology and synonyms. "
    "Return one query per line, no numbering.\n\nQuestion: {question}\nQueries:"
)
_HYDE_PROMPT = (
    "Write a short, factual paragraph that would directly answer the question, as if "
    "excerpted from a methods paper. Do not hedge.\n\nQuestion: {question}\nPassage:"
)


def transform_query(question: str, llm: LLMBackend, settings: Settings) -> QuerySpec:
    sub_queries: list[str] = []
    hyde_doc: str | None = None

    if settings.retrieval.use_multi_query:
        raw = llm.complete(_MULTIQUERY_PROMPT.format(question=question))
        sub_queries = _extract_queries(raw, question)

    if settings.retrieval.use_hyde:
        passage = llm.complete(_HYDE_PROMPT.format(question=question)).strip()
        if passage and "[mock-llm]" not in passage.lower():
            hyde_doc = passage

    return QuerySpec(original=question, sub_queries=sub_queries, hyde_doc=hyde_doc)


def _extract_queries(raw: str, original: str) -> list[str]:
    out: list[str] = []
    seen = {original.lower()}
    for line in raw.splitlines():
        line = re.sub(r"^[\-*\d.)\s]+", "", line.strip()).strip()
        if not line or len(line) < 5 or "[mock-llm]" in line.lower():
            continue
        if line.lower() in seen:
            continue
        seen.add(line.lower())
        out.append(line)
    return out[:3]
