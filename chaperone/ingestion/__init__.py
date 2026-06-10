"""Corpus ingestion: load sources, chunk them, index into the vector store."""

from chaperone.ingestion.loaders import load_sources, source_from_str
from chaperone.ingestion.pipeline import Ingestor

__all__ = ["Ingestor", "load_sources", "source_from_str"]
