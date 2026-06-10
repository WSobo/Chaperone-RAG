"""LLM backend contract.

Generation is pluggable so the rest of the system never imports torch or knows
which model is live. A backend only has to implement :meth:`complete`. Backends
that want to own answer structuring (the CPU mock does) may also expose
``draft_answer`` — the RAG chain prefers it when present and otherwise falls back
to ``complete`` + :func:`parse_draft`.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class DraftAnswer(BaseModel):
    """The LLM's contribution to an answer, before citations are resolved.

    ``used_markers`` are the ``[n]`` source numbers the model relied on; the RAG
    chain maps them back to real sources so citations cannot point at nonexistent
    context.
    """

    text: str
    used_markers: list[int] = Field(default_factory=list)
    grounded: bool = True
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal text-completion interface every backend implements."""

    name: str

    def complete(self, prompt: str, *, system: str | None = None) -> str: ...


def parse_draft(raw: str) -> DraftAnswer:
    """Tolerantly parse a model completion into a :class:`DraftAnswer`.

    Accepts a JSON object (possibly fenced or surrounded by prose); falls back to
    treating the whole string as the answer and scraping ``[n]`` markers from it.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            markers = [int(x) for x in data.get("used_markers", []) if _is_int(x)]
            return DraftAnswer(
                text=str(data.get("text", "")).strip() or raw.strip(),
                used_markers=markers,
                grounded=bool(data.get("grounded", True)),
                confidence=float(data.get("confidence", 0.5)),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    markers = sorted({int(x) for x in re.findall(r"\[(\d+)\]", raw)})
    text = raw.strip()
    return DraftAnswer(text=text, used_markers=markers, grounded=bool(text), confidence=0.4)


def _is_int(x: object) -> bool:
    return isinstance(x, int) or (isinstance(x, str) and x.strip().isdigit())
