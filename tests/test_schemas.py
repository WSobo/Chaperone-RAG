"""Schema contract tests (pydantic only — no heavy deps)."""

import pytest
from pydantic import ValidationError

from chaperone.schemas import Answer, Chunk, Citation, QuerySpec, SourceType


def test_chunk_short_source_with_and_without_page():
    c = Chunk(id="1", text="t", source_uri="paper.pdf", title="Paper", page=4)
    assert c.short_source() == "Paper (p.4)"
    c2 = Chunk(id="2", text="t", source_uri="paper.pdf", title="Paper")
    assert c2.short_source() == "Paper"


def test_answer_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Answer(question="q", text="a", made_up=True)


def test_citation_marker_must_be_positive():
    with pytest.raises(ValidationError):
        Citation(marker=0, source_uri="x")


def test_confidence_bounds_enforced():
    with pytest.raises(ValidationError):
        Answer(question="q", text="a", confidence=1.5)


def test_queryspec_dedup_and_original_first():
    qs = QuerySpec(original="design a binder", sub_queries=["make a binder", "make a binder", ""])
    assert qs.all_queries() == ["design a binder", "make a binder"]


def test_source_type_enum():
    c = Chunk(id="1", text="t", source_uri="u", source_type=SourceType.web)
    assert c.source_type is SourceType.web
