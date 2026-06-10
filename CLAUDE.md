# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Chaperone-RAG** is a domain-specialized Retrieval-Augmented Generation system for **protein design and engineering** (RFdiffusion, ProteinMPNN, AlphaFold/ESMFold, ESM, Rosetta, etc.). It grounds a locally-hosted open-weights LLM (target: Gemma) in a curated corpus of methods papers, tool docs, and structural data, and returns **cited, schema-validated answers**.

Design goals, in priority order:
1. **Grounding over fluency** — every claim traces to a retrieved source; the system reports when it cannot answer from context.
2. **Typed end-to-end** — Pydantic v2 models are the contract between every stage; LLM output is parsed into validated objects, never passed around as raw strings.
3. **Runs without a GPU** — the generation backend is pluggable. A deterministic CPU mock exercises the *entire* retrieval + citation pipeline so the system is demoable and testable anywhere; the Gemma backend swaps in on an HPC GPU node with no other code change.
4. **Evaluated, not vibes** — retrieval and answer quality are measured with a RAGAS harness against a golden set.

This repo is **not** an MCP server. Tool-protocol / MCP work lives in the separate `genesis-bio-mcp` repo. Chaperone-RAG is a self-contained RAG application and library.

## Commands

Dependencies are declared in `pyproject.toml`. **uv** is the supported workflow (plain pip still works). uv manages the `.venv` and a committed `uv.lock`:

```bash
uv sync                                    # core RAG (CPU, mock LLM) — runs everywhere
uv sync --extra dev                        # + pytest/ruff/mypy
uv sync --extra eval                       # + ragas/datasets
uv sync --extra gpu --torch-backend=auto   # + torch/transformers/accelerate (GPU node)
uv lock                                    # regenerate the lockfile after changing deps
# pip fallback: pip install -e ".[dev]"
```

`scripts/setup_env.sh` bootstraps the full GPU env on the cluster with uv (and builds google-deepmind/gemma from source until it's pip-installable).

CLI (Typer, entry point `chaperone`; prefix with `uv run` to use the project venv):

```bash
uv run chaperone ingest data/papers/       # ingest a dir / files / URLs into the store
uv run chaperone ingest https://…          # URLs are auto-detected (no flag needed)
uv run chaperone ask "How does RFdiffusion condition on a binding hotspot?"
uv run chaperone chat                      # interactive REPL
uv run chaperone eval                      # evaluation harness against the golden set
python main.py chat                        # back-compat shim → the same CLI
```

Run on the cluster: `./run_chaperone.sh` requests an interactive GPU node (`srun`, A5500), activates the uv `.venv`, sets `CHAPERONE_LLM__BACKEND=gemma`, and starts the REPL.

Tests / quality:

```bash
uv run pytest                              # full suite (CPU, mock backend — no GPU/network)
uv run pytest tests/test_rag_pipeline.py   # a single test file
uv run ruff check . && uv run mypy chaperone
```

## Configuration

All runtime config is a single Pydantic-Settings tree in [chaperone/settings.py](chaperone/settings.py), layered as: defaults → `configs/chaperone.yaml` → environment variables. Env vars use the `CHAPERONE_` prefix with `__` as the nesting delimiter, e.g. `CHAPERONE_LLM__BACKEND=mock`, `CHAPERONE_RETRIEVAL__TOP_K=8`, `CHAPERONE_RETRIEVAL__RERANK_TOP_N=4`. Never read `os.environ` or hard-code paths/model IDs in modules — add a field to the settings tree and read it from there. (The old code hard-coded a cluster cache path in two files; that is now `settings.paths.model_cache`.)

## Architecture

The system is a **library** (`chaperone/`) with thin entry points. Data flows in one direction and every boundary is a Pydantic model defined in [chaperone/schemas.py](chaperone/schemas.py).

```
ingest:  source → loaders → chunking → embeddings → Chroma (persisted)
query:   question
           → query_transform (multi-query + HyDE)        retrieval/query_transform.py
           → hybrid retrieve (dense ⊕ BM25, RRF/ensemble) retrieval/hybrid.py
           → cross-encoder rerank (top-k → top-n)         retrieval/rerank.py
           → RAG chain: numbered context → LLM            rag/chain.py
           → Answer{ text, citations[], confidence }      schemas.py  (validated)
```

**`schemas.py` — the contract.** Pydantic v2 models used everywhere: `IngestSource`, `Chunk`, `RetrievedChunk` (chunk + score + rank), `Citation` (source id + locator + quote), `Answer` (answer text + `list[Citation]` + grounded flag + confidence), `QuerySpec` (transformed sub-queries). The retrieval stack passes `RetrievedChunk`s; the RAG chain emits a validated `Answer`. If you change a stage's I/O, change the model here first.

**`llm/` — pluggable generation.** `base.LLMBackend` is a `Protocol` with `generate(prompt) -> str` and `structured(prompt, schema) -> BaseModel`. Implementations:
- `mock.MockLLM` — **default, CPU-only.** Deterministic, citation-aware: it performs extractive synthesis over the retrieved context and emits a well-formed `Answer` with real `[n]` citations. No weights, no network. This is what makes the demo and the test suite real rather than stubbed.
- `gemma.GemmaLLM` — HuggingFace `transformers` backend (bf16, `device_map="auto"`), lazy-imports torch so importing the package never requires a GPU. Uses the model's chat template and constrained/structured decoding for the `Answer` schema.
- `factory.get_llm(settings)` selects the backend from `settings.llm.backend`. **Add a backend by implementing the Protocol and registering it here — nothing else in the codebase knows which LLM is live.**

**`retrieval/` — the SOTA core.** This is where the engineering signal is:
- `embeddings.py` — sentence-transformers embeddings; default `BAAI/bge-small-en-v1.5`, configurable to a domain model (e.g. PubMedBERT) via settings.
- `vectorstore.py` — Chroma wrapper (persisted to `settings.paths.vector_db`), dense similarity.
- `hybrid.py` — combines dense vector search with **BM25** lexical search via explicit **reciprocal-rank fusion** (first-party: a thin `rank_bm25` wrapper + the RRF, no langchain-community retriever). Lexical recall matters here because protein-design queries are dense with exact tokens (gene names, PDB IDs, "ProteinMPNN", "ipTM").
- `rerank.py` — a `BaseDocumentCompressor` wrapping a **cross-encoder** (`BAAI/bge-reranker-base`) that re-scores the hybrid candidates and keeps the top-n. Biggest single quality lever.
- `query_transform.py` — **multi-query** expansion and **HyDE** (hypothetical-document embedding) as LCEL runnables, to fix lexical-gap recall failures.
- `retriever.py` — composes the above into one `Retriever` (transform → hybrid → rerank) returning `list[RetrievedChunk]`.

**`rag/` — the answer chain.** `chain.py` is an **LCEL** pipeline: retrieve → render numbered context blocks → prompt (`prompts.py`) → LLM → parse into `Answer`. Citations are bound to the numbered sources so they're verifiable, not hallucinated. The chain enforces the "say so when unsupported" behavior.

**`agent/` — optional agentic mode (LangGraph).** A `StateGraph` for multi-hop questions that need live tools: `transform → retrieve → grade → {generate | call_tool → retrieve}`. The plain `rag/` chain is the default fast path; the agent is for questions a single retrieval pass can't satisfy.

**`tools/` — typed bio tools.** LangChain tools with Pydantic argument schemas: `rcsb` (PDB metadata/coordinate download), `literature` (arXiv / web search), `slurm` (write + `sbatch` HPC jobs), `sandbox` (timeboxed Python execution for on-the-fly BioPython work). The sandbox runs untrusted generated code — keep it constrained (timeout, workspace dir) and treat it as a security boundary.

**`eval/` — measurement.** `golden_set.py` holds protein-design Q/A/ground-truth-context fixtures; `ragas_eval.py` scores faithfulness, answer relevancy, and context precision/recall. Runs against the mock backend in CI so retrieval regressions are caught without a GPU.

**`utils/logger.py`** — shared `rich` logger; import `logger` from here rather than using `print`.

## Conventions

- **Pydantic v2 only.** Use `model_config = ConfigDict(...)`, `field_validator`/`model_validator`, `model_dump()`. No v1 `.dict()`/`class Config`/`@validator`.
- **Settings, not globals.** New config → a field on the settings tree; read it via the passed-in `settings`, don't reach into `os.environ`.
- **Boundaries are models.** A function that crosses a pipeline stage takes and returns a `schemas.py` model, not a dict or tuple.
- **LangChain composition is LCEL.** Prefer `Runnable` pipes (`|`) over imperative glue; use LangGraph only for the stateful agent loop.
- **The mock backend must stay first-class.** Every feature has to work (and be tested) under `MockLLM` on CPU. If a change only works with real Gemma weights, it's not done.
- **Add a tool:** define its Pydantic args + `@tool` in `tools/`, then register it in the agent's tool list — that's the only wiring point.
- Runtime artifacts (`data/`, vector DB, `model_cache/`, the cloned `gemma/`, weights) are gitignored and created on demand.

## Status

The architecture above is implemented end-to-end. The legacy single-file demo
(`engine.py`/`memory.py`, manual context concatenation) has been removed — there is
one pipeline, the typed one. The full CPU path (mock LLM + hybrid retrieval + rerank +
cited `Answer`) is covered by `tests/`, including an offline end-to-end test that uses
deterministic fake embeddings (no GPU, no network). The Gemma backend and RAGAS
generation metrics are wired but exercised only where their extra deps / GPU are
available. When extending, add to the typed pipeline; don't reintroduce string-passing
between stages.
