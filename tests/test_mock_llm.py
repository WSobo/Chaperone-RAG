"""MockLLM extractive-synthesis behavior (no heavy deps)."""

from chaperone.llm.base import parse_draft
from chaperone.llm.mock import MockLLM
from chaperone.schemas import Chunk, RetrievedChunk, SourceType


def _rc(i, text, title=None, page=None):
    return RetrievedChunk(
        chunk=Chunk(id=f"c{i}", text=text, source_uri=f"s{i}", source_type=SourceType.pdf,
                    title=title, page=page),
        score=1.0 / (i + 1),
        rank=i,
    )


def test_draft_cites_only_matching_source():
    contexts = [
        _rc(0, "RFdiffusion conditions on a binding hotspot to design de novo binders.", "RFdiffusion"),
        _rc(1, "Unrelated text about chromatography buffers and columns.", "Other"),
    ]
    draft = MockLLM().draft_answer("How does RFdiffusion use a binding hotspot?", contexts)
    assert draft.grounded
    assert draft.used_markers == [1]
    assert "[1]" in draft.text


def test_empty_context_is_ungrounded():
    draft = MockLLM().draft_answer("anything", [])
    assert draft.grounded is False
    assert draft.used_markers == []
    assert draft.confidence == 0.0


def test_parse_draft_handles_json_and_prose():
    j = parse_draft('{"text": "ok [2]", "used_markers": [2], "grounded": true, "confidence": 0.8}')
    assert j.used_markers == [2] and j.confidence == 0.8
    p = parse_draft("plain answer citing [1] and [3]")
    assert p.used_markers == [1, 3]
