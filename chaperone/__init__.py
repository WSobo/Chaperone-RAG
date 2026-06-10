"""Chaperone-RAG: cited, grounded RAG for protein design and engineering."""

import os as _os

__version__ = "0.1.0"

# Identify our web requests politely (also silences langchain's USER_AGENT warning).
_os.environ.setdefault("USER_AGENT", f"Chaperone-RAG/{__version__}")
