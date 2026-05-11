"""EmbeddingGroundingDetector + cosine_similarity tests (fake embedder, no torch)."""

from __future__ import annotations

import math

import pytest

from soweak import (
    Boundary,
    Context,
    EmbeddingGroundingDetector,
    OwaspCategory,
    Payload,
    Severity,
    cosine_similarity,
)
from soweak.grounding import RETRIEVED_DOCS_KEY, RETRIEVED_TEXT_KEY


# ---------------- cosine_similarity ----------------


def test_cosine_identical_vectors():
    assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        cosine_similarity([1.0], [1.0, 0.0])


# ---------------- EmbeddingGroundingDetector ----------------


def _fake_embedder(mapping: dict[str, list[float]]):
    """Deterministic embedder: returns the vector mapped to each input string,
    or a zero-vector fallback when missing."""

    def _embed(texts: list[str]) -> list[list[float]]:
        return [mapping.get(t, [0.0, 0.0, 0.0]) for t in texts]

    return _embed


def test_embedding_grounding_passes_similar_content():
    embedder = _fake_embedder(
        {
            "context text": [1.0, 0.0, 0.0],
            "this sentence is grounded ok": [0.95, 0.05, 0.0],
        }
    )
    det = EmbeddingGroundingDetector(embedder=embedder, threshold=0.6, min_sentence_tokens=3)
    ctx = Context(metadata={RETRIEVED_TEXT_KEY: "context text"})
    sigs = list(
        det.inspect(
            Payload(Boundary.OUTPUT, text="this sentence is grounded ok"),
            ctx,
        )
    )
    assert sigs == []


def test_embedding_grounding_flags_dissimilar_content():
    embedder = _fake_embedder(
        {
            "context about cats": [1.0, 0.0, 0.0],
            "wild claim about dolphins on mars": [0.0, 1.0, 0.0],
        }
    )
    det = EmbeddingGroundingDetector(embedder=embedder, threshold=0.5, min_sentence_tokens=3)
    ctx = Context(metadata={RETRIEVED_TEXT_KEY: "context about cats"})
    sigs = list(
        det.inspect(
            Payload(Boundary.OUTPUT, text="wild claim about dolphins on mars"),
            ctx,
        )
    )
    assert len(sigs) == 1
    assert sigs[0].category is OwaspCategory.LLM09_MISINFORMATION
    assert sigs[0].metadata["similarity"] < 0.5


def test_embedding_grounding_skipped_without_retrieval():
    embedder = _fake_embedder({})
    det = EmbeddingGroundingDetector(embedder=embedder)
    sigs = list(
        det.inspect(Payload(Boundary.OUTPUT, text="anything goes here"), Context())
    )
    assert sigs == []


def test_embedding_grounding_skips_short_sentences():
    called: list[list[str]] = []

    def embedder(texts: list[str]) -> list[list[float]]:
        called.append(texts)
        return [[1.0, 0.0, 0.0]] * len(texts)

    det = EmbeddingGroundingDetector(embedder=embedder, min_sentence_tokens=5)
    ctx = Context(metadata={RETRIEVED_TEXT_KEY: "some context"})
    list(det.inspect(Payload(Boundary.OUTPUT, text="Yes."), ctx))
    # Sentence had only 1 token, no embedder call made.
    assert called == []


def test_embedding_grounding_reads_from_documents_list():
    embedder = _fake_embedder(
        {
            "doc one\n\ndoc two": [1.0, 0.0, 0.0],
            "this matches doc content": [0.95, 0.05, 0.0],
        }
    )
    det = EmbeddingGroundingDetector(embedder=embedder, threshold=0.5, min_sentence_tokens=3)
    ctx = Context(
        metadata={
            RETRIEVED_DOCS_KEY: [{"text": "doc one"}, {"page_content": "doc two"}]
        }
    )
    sigs = list(
        det.inspect(
            Payload(Boundary.OUTPUT, text="this matches doc content"),
            ctx,
        )
    )
    assert sigs == []


def test_embedding_grounding_threshold_validation():
    embedder = _fake_embedder({})
    with pytest.raises(ValueError, match="threshold"):
        EmbeddingGroundingDetector(embedder=embedder, threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        EmbeddingGroundingDetector(embedder=embedder, threshold=1.5)


def test_embedding_grounding_validates_embedder_output():
    """If the embedder returns the wrong number of vectors, the detector errors
    out clearly rather than silently mis-aligning."""

    def bad_embedder(texts: list[str]) -> list[list[float]]:
        return [[1.0]]  # too few

    det = EmbeddingGroundingDetector(embedder=bad_embedder, threshold=0.5, min_sentence_tokens=3)
    ctx = Context(metadata={RETRIEVED_TEXT_KEY: "context"})
    with pytest.raises(RuntimeError, match="wrong number of vectors"):
        list(det.inspect(Payload(Boundary.OUTPUT, text="some long enough sentence here"), ctx))


def test_embedding_grounding_multiple_sentences_emits_per_sentence():
    """Each ungrounded sentence yields its own signal."""
    # NB: the sentence splitter keeps trailing punctuation, so the keys here
    # include the period.
    embedder = _fake_embedder(
        {
            "context A": [1.0, 0.0, 0.0],
            "this is grounded close to A.": [0.95, 0.05, 0.0],
            "wildly off in a different direction.": [0.0, 0.0, 1.0],
            "also unrelated topic completely separate.": [-1.0, 0.0, 0.0],
        }
    )
    det = EmbeddingGroundingDetector(embedder=embedder, threshold=0.5, min_sentence_tokens=3)
    ctx = Context(metadata={RETRIEVED_TEXT_KEY: "context A"})
    text = (
        "this is grounded close to A. "
        "wildly off in a different direction. "
        "also unrelated topic completely separate."
    )
    sigs = list(det.inspect(Payload(Boundary.OUTPUT, text=text), ctx))
    assert len(sigs) == 2


def test_embedding_grounding_default_boundary_is_output():
    det = EmbeddingGroundingDetector(embedder=_fake_embedder({}))
    assert det.boundaries == (Boundary.OUTPUT,)


def test_embedding_grounding_default_category_is_llm09():
    det = EmbeddingGroundingDetector(embedder=_fake_embedder({}))
    assert det.category is OwaspCategory.LLM09_MISINFORMATION


def test_sentence_transformer_factory_raises_without_extras():
    """Importing the factory works, but calling it without sentence-transformers fails clearly."""
    import sys

    from soweak.embeddings import sentence_transformer_embedder

    if "sentence_transformers" in sys.modules:
        pytest.skip("sentence-transformers is installed; this test exercises the missing-extras path")
    with pytest.raises(ImportError, match=r"soweak\[embeddings\]"):
        sentence_transformer_embedder()
