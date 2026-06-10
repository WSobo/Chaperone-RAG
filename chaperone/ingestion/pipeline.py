"""Ingestion orchestration: load → split → index."""

from __future__ import annotations

from chaperone.ingestion.chunking import split_documents
from chaperone.ingestion.loaders import load_sources, source_from_str
from chaperone.retrieval.vectorstore import VectorStore
from chaperone.schemas import IngestSource
from chaperone.settings import Settings
from chaperone.utils.logger import logger


class Ingestor:
    """Loads sources, chunks them, and writes them to the vector store."""

    def __init__(self, settings: Settings, vector_store: VectorStore) -> None:
        self._settings = settings
        self._store = vector_store

    def ingest(self, sources: list[IngestSource]) -> int:
        docs = load_sources(sources)
        if not docs:
            logger.info("Nothing to ingest.")
            return 0
        chunks = split_documents(docs, self._settings)
        return self._store.add(chunks)

    def ingest_paths(self, raw_inputs: list[str]) -> int:
        """Ingest a mixed list of file paths, directories, and URLs."""
        return self.ingest([source_from_str(r) for r in raw_inputs])
