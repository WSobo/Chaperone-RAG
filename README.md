# Chaperone-RAG

**Domain-specialized, citation-grounded Retrieval-Augmented Generation for protein design and engineering.**

Chaperone-RAG grounds a locally-hosted open-weights LLM (target: Gemma) in a curated
corpus of protein-design methods — RFdiffusion, ProteinMPNN, AlphaFold/ESMFold, ESM,
Rosetta — and returns **cited, schema-validated answers** instead of confident prose.
It is built to run on an HPC GPU node, but the entire retrieval + citation pipeline
runs **on CPU with a deterministic mock backend**, so it's demoable and testable
anywhere.

> Not an MCP server. Tool-protocol / MCP work lives in a separate `genesis-bio-mcp`
> repo; this is a self-contained RAG application and library.

## Why it's built this way

| Principle | How it shows up in the code |
|---|---|
| **Grounding over fluency** | Every answer is an `Answer` model with `Citation`s whose markers are validated against the *actually retrieved* sources — the model cannot cite a source it wasn't shown. Unsupported questions return `grounded=False`, not a guess. |
| **Typed end-to-end** | Pydantic v2 models are the contract between every stage (`schemas.py`). The LLM's output is *parsed into a validated object*, never passed around as a string. |
| **Runs without a GPU** | Generation is a pluggable backend behind a `Protocol`. `MockLLM` does extractive, citation-aware synthesis on CPU; `GemmaLLM` swaps in on a GPU node with no other change. |
| **Evaluated, not vibes** | A RAGAS + deterministic-retrieval harness scores the pipeline against a labeled golden set. |

## Architecture

```
ingest:  PDFs / URLs ──> loaders ──> chunking ──> embeddings ──> Chroma (persisted)

query:   question
   │
   ├─ query transform     multi-query expansion + HyDE            retrieval/query_transform.py
   ├─ hybrid retrieve      dense (BGE) ⊕ BM25, reciprocal-rank     retrieval/hybrid.py
   ├─ cross-encoder rerank  top-k ─> top-n (BGE-reranker)          retrieval/rerank.py
   ├─ generate             numbered context ─> LLM                 rag/chain.py  (LCEL)
   └─ Answer{ text, citations[], grounded, confidence }           schemas.py    (validated)
```

The four levers a production RAG reviewer looks for — **hybrid retrieval**,
**cross-encoder reranking**, **query transformation**, and **grounded structured
output** — are each their own module, wired together by `retrieval/retriever.py` and
`rag/chain.py`. An optional **LangGraph** corrective-RAG agent (`agent/graph.py`) adds
a self-correction loop that falls back to web search when local recall is weak.

## Quickstart (CPU, no GPU, no API keys)

Uses [uv](https://docs.astral.sh/uv/) (plain pip works too — see below).

```bash
uv sync                      # core deps; CPU + mock backend

# Ingest some sources (PDFs, a directory, or URLs)
uv run chaperone ingest data/papers/
uv run chaperone ingest https://raw.githubusercontent.com/RosettaCommons/RFdiffusion/main/README.md

# Ask — returns a cited answer from the mock backend
uv run chaperone ask "How does RFdiffusion condition on a binding hotspot?"

# Interactive REPL / settings / evaluation
uv run chaperone chat
uv run chaperone info
uv run chaperone eval
```

> Plain pip: `pip install -e . && chaperone ask "..."` — same CLI, no uv required.

Example (mock backend):

```
╭─ Chaperone · grounded · conf 0.67 ───────────────────────────────╮
│ Based on the retrieved sources: It can condition on a binding     │
│ hotspot to design binders. [1]                                    │
╰───────────────────────────────────────────────────────────────────╯
 Sources
  #  Source         Where
  1  RFdiffusion    p.3
```

## Configuration

All runtime config is one typed Pydantic-Settings tree (`chaperone/settings.py`),
layered **defaults → `configs/chaperone.yaml` → `CHAPERONE_*` env vars**:

```bash
CHAPERONE_LLM__BACKEND=gemma          # mock | gemma
CHAPERONE_RETRIEVAL__TOP_K=20         # hybrid candidates
CHAPERONE_RETRIEVAL__RERANK_TOP_N=6   # kept after rerank
CHAPERONE_RETRIEVAL__USE_HYDE=true
```

## Running the real model on HPC

```bash
bash scripts/setup_env.sh                # installs uv, syncs the GPU env (+ gemma from source)
uv run python install_model_weights.py   # pre-fetch weights (optional; needs huggingface-cli login)
./run_chaperone.sh                       # srun an interactive GPU node, launch with backend=gemma
```

The model id and cache directory are settings fields — nothing is hard-coded.

## Evaluation

`chaperone eval` builds an ephemeral corpus from a bundled protein-design golden set
and reports:

- **Retrieval** (always, deterministic, CI-safe): hit-rate, MRR, grounded-rate.
- **Generation** (optional, needs `.[eval]` + an evaluator LLM): RAGAS faithfulness,
  answer relevancy, context precision/recall.

## Project layout

```
chaperone/
  settings.py      schemas.py      app.py        cli.py
  llm/             base · mock (CPU) · gemma (GPU) · factory
  ingestion/       loaders · chunking · pipeline
  retrieval/       embeddings · vectorstore · query_transform · hybrid · rerank · retriever
  rag/             prompts · chain (LCEL → cited Answer)
  agent/           graph (LangGraph corrective-RAG)
  tools/           rcsb · literature · slurm · sandbox  (typed @tool)
  eval/            golden_set · harness (RAGAS + retrieval metrics)
tests/             schemas · settings · mock_llm · rag_pipeline (offline e2e)
```

## Testing

```bash
uv sync --extra dev
uv run pytest          # mock backend, offline; the e2e test uses fake embeddings
uv run ruff check . && uv run mypy chaperone
```

See [CLAUDE.md](CLAUDE.md) for the architecture/conventions reference.
