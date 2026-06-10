"""HuggingFace Gemma backend for local generation on a GPU node.

torch/transformers are imported lazily inside ``__init__`` so that importing the
package (and running the whole mock-backed pipeline) never requires a GPU or the
``[gpu]`` extra. Install with ``pip install -e ".[gpu]"`` and select via
``CHAPERONE_LLM__BACKEND=gemma``.
"""

from __future__ import annotations

from chaperone.settings import Settings
from chaperone.utils.logger import logger


class GemmaLLM:
    """Open-weights Gemma via ``transformers``, bf16, sharded across visible GPUs."""

    name = "gemma"

    def __init__(self, settings: Settings) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        cfg = settings.llm
        cache_dir = str(settings.paths.model_cache)
        logger.info(f"Loading {cfg.model_id} (bf16) from cache {cache_dir} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
        self._pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature or None,
            return_full_text=False,
        )
        logger.info("Gemma backend ready.")

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        # Gemma's chat template has no system role, so fold any system prompt into
        # the user turn rather than passing an unsupported role.
        content = f"{system.strip()}\n\n{prompt}" if system else prompt
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out = self._pipe(rendered)[0]["generated_text"]
        return out.strip()
