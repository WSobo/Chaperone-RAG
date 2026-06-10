"""Settings layering: defaults, YAML overlay, env-var override."""

from chaperone.settings import Settings


def test_defaults_and_yaml_overlay():
    s = Settings()
    assert s.llm.backend == "mock"  # from configs/chaperone.yaml
    assert s.retrieval.top_k == 12
    assert s.retrieval.use_reranker is True


def test_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("CHAPERONE_LLM__BACKEND", "gemma")
    monkeypatch.setenv("CHAPERONE_RETRIEVAL__RERANK_TOP_N", "9")
    s = Settings()
    assert s.llm.backend == "gemma"
    assert s.retrieval.rerank_top_n == 9


def test_ensure_dirs_is_idempotent(tmp_path):
    s = Settings()
    s.paths.data_dir = tmp_path / "data"
    s.paths.papers_dir = tmp_path / "data" / "papers"
    s.paths.vector_db = tmp_path / "data" / "db"
    s.paths.sandbox_dir = tmp_path / "data" / "sandbox"
    s.paths.pdb_dir = tmp_path / "data" / "pdb"
    s.paths.scripts_dir = tmp_path / "scripts"
    s.ensure_dirs()
    s.ensure_dirs()  # second call must not raise
    assert (tmp_path / "data" / "papers").is_dir()
