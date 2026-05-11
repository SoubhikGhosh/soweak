"""LLM09 grounding and citation tests."""

from __future__ import annotations

import pytest

from soweak import (
    Boundary,
    CitationRequiredDetector,
    Context,
    GroundingDetector,
    OwaspCategory,
    Payload,
    Severity,
)
from soweak.grounding import RETRIEVED_DOCS_KEY, RETRIEVED_TEXT_KEY


# ---------------- CitationRequiredDetector ----------------


def test_citation_fires_on_long_uncited_output():
    det = CitationRequiredDetector(min_chars=100)
    text = "x " * 80  # >100 chars, no citation
    sigs = list(det.inspect(Payload(Boundary.OUTPUT, text=text), Context()))
    assert len(sigs) == 1
    assert sigs[0].category is OwaspCategory.LLM09_MISINFORMATION


def test_citation_allows_short_output():
    det = CitationRequiredDetector(min_chars=200)
    sigs = list(det.inspect(Payload(Boundary.OUTPUT, text="short answer"), Context()))
    assert sigs == []


def test_citation_recognises_bracket_refs():
    det = CitationRequiredDetector(min_chars=10)
    text = "The capital is Paris [wiki-fr-paris]. " * 5
    assert list(det.inspect(Payload(Boundary.OUTPUT, text=text), Context())) == []


def test_citation_recognises_numeric_refs():
    det = CitationRequiredDetector(min_chars=10)
    text = "See reference (1). " * 10
    assert list(det.inspect(Payload(Boundary.OUTPUT, text=text), Context())) == []


def test_citation_custom_severity():
    det = CitationRequiredDetector(min_chars=10, severity=Severity.HIGH)
    sigs = list(det.inspect(Payload(Boundary.OUTPUT, text="x " * 20), Context()))
    assert sigs[0].severity is Severity.HIGH


def test_citation_validates_min_chars():
    with pytest.raises(ValueError):
        CitationRequiredDetector(min_chars=0)


# ---------------- GroundingDetector ----------------


def test_grounding_passes_when_overlap_high():
    det = GroundingDetector(min_overlap=0.3)
    ctx = Context(
        metadata={
            RETRIEVED_TEXT_KEY: (
                "The Eiffel Tower is located in Paris, France. "
                "It was completed in 1889 by Gustave Eiffel."
            )
        }
    )
    output = "The Eiffel Tower is located in Paris and was completed in 1889."
    assert list(det.inspect(Payload(Boundary.OUTPUT, text=output), ctx)) == []


def test_grounding_flags_ungrounded_sentence():
    det = GroundingDetector(min_overlap=0.5, min_sentence_tokens=3)
    ctx = Context(
        metadata={
            RETRIEVED_TEXT_KEY: (
                "Bananas are yellow tropical fruits. They grow in clusters."
            )
        }
    )
    output = (
        "Bananas are yellow tropical fruits. "
        "The Roman Empire fell because of inflation and barbarian invasions."
    )
    sigs = list(det.inspect(Payload(Boundary.OUTPUT, text=output), ctx))
    assert sigs
    assert "Roman" in (sigs[0].matched_text or "")


def test_grounding_skipped_without_retrieval_context():
    det = GroundingDetector()
    assert (
        list(det.inspect(Payload(Boundary.OUTPUT, text="anything goes"), Context()))
        == []
    )


def test_grounding_reads_from_documents_list():
    det = GroundingDetector(min_overlap=0.3)
    ctx = Context(
        metadata={
            RETRIEVED_DOCS_KEY: [
                {"text": "Mars has two moons, Phobos and Deimos."},
                {"page_content": "Mars is the fourth planet from the Sun."},
            ]
        }
    )
    output = "Mars has two moons, Phobos and Deimos."
    assert list(det.inspect(Payload(Boundary.OUTPUT, text=output), ctx)) == []


def test_grounding_skips_short_sentences():
    det = GroundingDetector(min_overlap=0.5, min_sentence_tokens=5)
    ctx = Context(metadata={RETRIEVED_TEXT_KEY: "Some unrelated context."})
    output = "Yes."  # tokens < threshold
    assert list(det.inspect(Payload(Boundary.OUTPUT, text=output), ctx)) == []


def test_grounding_validates_min_overlap():
    with pytest.raises(ValueError):
        GroundingDetector(min_overlap=0.0)
    with pytest.raises(ValueError):
        GroundingDetector(min_overlap=1.5)
