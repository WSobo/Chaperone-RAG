"""Evaluation: deterministic retrieval metrics + optional RAGAS generation metrics."""

from chaperone.eval.golden_set import GOLDEN, GoldenItem
from chaperone.eval.harness import EvalReport, ItemResult, run_eval

__all__ = ["EvalReport", "GOLDEN", "GoldenItem", "ItemResult", "run_eval"]
