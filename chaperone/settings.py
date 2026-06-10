"""Typed, layered configuration — the single source of runtime truth.

Priority (highest first): constructor kwargs > ``CHAPERONE_*`` env vars >
``configs/chaperone.yaml`` > field defaults. Nothing in the codebase reads
``os.environ`` or hard-codes a path/model id directly; add a field here instead.

    from chaperone.settings import get_settings
    settings = get_settings()
    settings.retrieval.rerank_top_n   # -> int

Override at the shell:  CHAPERONE_LLM__BACKEND=gemma  CHAPERONE_RETRIEVAL__TOP_K=20
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class PathSettings(BaseModel):
    """Filesystem layout. All runtime dirs are created on demand."""

    data_dir: Path = Path("data")
    papers_dir: Path = Path("data/papers")
    vector_db: Path = Path("data/chroma_db")
    sandbox_dir: Path = Path("data/sandbox")
    pdb_dir: Path = Path("data/pdb_files")
    scripts_dir: Path = Path("scripts")
    skills_dir: Path = Path("configs/skills")  # tool manifests (*.yaml)
    runs_dir: Path = Path("data/runs")  # job run records / provenance
    # Was hard-coded to a cluster path in two modules; now one configurable field.
    model_cache: Path = Path("model_cache")


class LLMSettings(BaseModel):
    backend: Literal["mock", "gemma"] = "mock"
    model_id: str = "google/gemma-2-9b-it"
    max_new_tokens: int = 1024
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class EmbeddingSettings(BaseModel):
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"  # "cuda" on a GPU node


class RetrievalSettings(BaseModel):
    collection_name: str = "chaperone_docs"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = Field(default=12, ge=1, description="Candidates returned by hybrid search.")
    rerank_top_n: int = Field(default=4, ge=1, description="Kept after cross-encoder rerank.")
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    use_multi_query: bool = True
    use_hyde: bool = False


class JobsSettings(BaseModel):
    # "local" runs job scripts as subprocesses (CPU/dev/CI); "slurm" submits via sbatch.
    scheduler: Literal["slurm", "local"] = "local"
    poll_interval: float = Field(default=2.0, gt=0, description="Seconds between status polls.")


class Settings(BaseSettings):
    """Root settings tree. Construct via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="CHAPERONE_",
        env_nested_delimiter="__",
        yaml_file="configs/chaperone.yaml",
        extra="ignore",
    )

    paths: PathSettings = Field(default_factory=PathSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    jobs: JobsSettings = Field(default_factory=JobsSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Insert YAML below env so env vars win, but above field defaults.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    def ensure_dirs(self) -> None:
        """Create the runtime directories this run will write to."""
        for p in (
            self.paths.data_dir,
            self.paths.papers_dir,
            self.paths.vector_db,
            self.paths.sandbox_dir,
            self.paths.pdb_dir,
            self.paths.scripts_dir,
            self.paths.runs_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
