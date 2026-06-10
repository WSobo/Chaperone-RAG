"""Embedding model construction.

Default is BGE-small (strong, light, English). For a domain-tuned corpus you can
point ``embedding.model_name`` at a biomedical encoder (e.g. PubMedBERT) without
touching code.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from chaperone.settings import Settings


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    cfg = settings.embedding
    return HuggingFaceEmbeddings(
        model_name=cfg.model_name,
        model_kwargs={"device": cfg.device},
        # Normalized embeddings + cosine play well with BGE and with reranking.
        encode_kwargs={"normalize_embeddings": True},
    )
