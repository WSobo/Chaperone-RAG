"""Select the live LLM backend from settings.

This is the *only* place that knows which concrete backend exists. Add a new one
by implementing :class:`~chaperone.llm.base.LLMBackend` and registering it here.
"""

from __future__ import annotations

from chaperone.llm.base import LLMBackend
from chaperone.settings import Settings, get_settings


def get_llm(settings: Settings | None = None) -> LLMBackend:
    settings = settings or get_settings()
    backend = settings.llm.backend

    if backend == "mock":
        from chaperone.llm.mock import MockLLM

        return MockLLM()

    if backend == "gemma":
        from chaperone.llm.gemma import GemmaLLM  # lazy: pulls in torch only here

        return GemmaLLM(settings)

    raise ValueError(f"Unknown LLM backend: {backend!r} (expected 'mock' or 'gemma').")
