"""Pre-fetch the configured LLM weights into the model cache (run on a GPU node).

Optional convenience — the Gemma backend also downloads on first use. The model id
and cache directory come from settings (CHAPERONE_LLM__MODEL_ID / paths.model_cache),
so there are no hard-coded paths here.

Note: Gemma checkpoints are gated on Hugging Face; run `huggingface-cli login` first.
"""

from __future__ import annotations

import os


def main() -> None:
    from chaperone.settings import get_settings

    settings = get_settings()
    cache = str(settings.paths.model_cache)
    os.environ.setdefault("HF_HOME", cache)

    from huggingface_hub import snapshot_download

    path = snapshot_download(settings.llm.model_id, cache_dir=cache)
    print(f"Cached {settings.llm.model_id} -> {path}")


if __name__ == "__main__":
    main()
