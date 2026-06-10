#!/bin/bash
# Bootstrap the Chaperone-RAG environment on the HPC cluster with uv.
# Usage:  bash scripts/setup_env.sh      (run on a GPU node with the NVIDIA driver visible)
set -euo pipefail

# 1) Install uv if it isn't on PATH (user-local, no root required).
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2) Create the project venv (Python pinned by .python-version) and sync deps.
#    UV_TORCH_BACKEND=auto selects the CUDA torch wheel matching this node.
echo "==> Syncing dependencies (gpu + eval + dev)"
UV_TORCH_BACKEND=auto uv sync --extra gpu --extra eval --extra dev

# 3) Build the latest Gemma from source (remove once it's on PyPI; then add to [gpu]).
if [ ! -d gemma ]; then
  echo "==> Cloning google-deepmind/gemma"
  git clone https://github.com/google-deepmind/gemma.git
fi
uv pip install -e ./gemma
uv pip install "jax[cuda12]"

echo "==> Done."
echo "    Weights:  uv run python install_model_weights.py   (after: huggingface-cli login)"
echo "    Launch:   CHAPERONE_LLM__BACKEND=gemma uv run chaperone chat"

# --- Fallback: uv inside an existing conda env --------------------------------
# If your cluster mandates conda for the CUDA base, use this instead of steps 1-3:
#   conda create -n chaperone_env python=3.12 -y && conda activate chaperone_env
#   pip install uv
#   UV_TORCH_BACKEND=auto uv pip install -e ".[gpu,eval,dev]"
#   uv pip install -e ./gemma "jax[cuda12]"
