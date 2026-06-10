"""Pluggable LLM backends (mock for CPU, Gemma for GPU)."""

from chaperone.llm.base import DraftAnswer, LLMBackend, parse_draft
from chaperone.llm.factory import get_llm
from chaperone.llm.mock import MockLLM

__all__ = ["DraftAnswer", "LLMBackend", "MockLLM", "get_llm", "parse_draft"]
