"""Embedding-based grounding (LLM09) and the sentence-transformer factory.

The lexical-overlap :class:`~soweak.grounding.GroundingDetector` is fast
but fragile to paraphrase. :class:`EmbeddingGroundingDetector` computes
cosine similarity between each output sentence and the retrieval context,
catching ungrounded claims even when they share vocabulary with the
source.

Like every other ML integration in soweak, the detector itself is
dependency-free: it takes any ``Callable[[list[str]], list[list[float]]]``
that turns a batch of strings into vectors. The bundled
:func:`sentence_transformer_embedder` factory requires
``pip install soweak[embeddings]``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Iterator

try:  # numpy is optional; we fall back to pure Python.
    import numpy as _np  # type: ignore[import-not-found]

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - exercised only when numpy is absent
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

from soweak.core.detector import Detector, Signal
from soweak.core.types import Boundary, Context, OwaspCategory, Payload, Severity
from soweak.grounding import (
    RETRIEVED_DOCS_KEY,
    RETRIEVED_TEXT_KEY,
    gather_retrieval,
    split_sentences,
)


# ---------------------------------------------------------------------------
# Embedder protocol type alias
# ---------------------------------------------------------------------------


#: Callable that turns a batch of strings into a batch of vectors. Vectors
#: are plain Python lists of floats — soweak doesn't take a hard dependency
#: on numpy. Sentence-transformers' ``.encode(...).tolist()`` matches this
#: shape directly.
Embedder = Callable[[list[str]], list[list[float]]]


# ---------------------------------------------------------------------------
# Cosine helper
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Uses numpy when available (O(n) C loop); falls back to a pure-Python
    implementation otherwise. Returns 0.0 if either vector is the zero
    vector.
    """
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    if _HAS_NUMPY:
        va = _np.asarray(a, dtype=_np.float64)
        vb = _np.asarray(b, dtype=_np.float64)
        na = float(_np.linalg.norm(va))
        nb = float(_np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(_np.dot(va, vb) / (na * nb))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ---------------------------------------------------------------------------
# EmbeddingGroundingDetector
# ---------------------------------------------------------------------------


class EmbeddingGroundingDetector(Detector):
    """Flag output sentences whose cosine similarity against the retrieval
    context falls below ``threshold``.

    Reads retrieval text from ``ctx.metadata["retrieved_text"]`` (a string)
    or ``ctx.metadata["retrieved_documents"]`` (a list of doc-shaped
    values). Splits the output into sentences, embeds the retrieval
    context plus all sentences in a single batch, and emits a signal per
    sentence whose similarity is below threshold.

    Sentences shorter than ``min_sentence_tokens`` whitespace tokens are
    skipped (too short to ground reliably).

    Honest framing: high similarity ≠ truth. A plausible fabrication that
    paraphrases the source closely will pass. Treat low-similarity signals
    as "look at this", not "wrong".
    """

    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.55,
        min_sentence_tokens: int = 4,
        retrieval_keys: tuple[str, ...] = (RETRIEVED_TEXT_KEY, RETRIEVED_DOCS_KEY),
        severity: Severity = Severity.MEDIUM,
        name: str = "embedding-grounding",
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if min_sentence_tokens < 1:
            raise ValueError("min_sentence_tokens must be positive")
        self._embedder = embedder
        self._threshold = threshold
        self._min_sentence_tokens = min_sentence_tokens
        self._retrieval_keys = retrieval_keys
        self._severity = severity
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> OwaspCategory:
        return OwaspCategory.LLM09_MISINFORMATION

    @property
    def boundaries(self) -> tuple[Boundary, ...]:
        return (Boundary.OUTPUT,)

    def inspect(self, payload: Payload, ctx: Context) -> Iterable[Signal]:
        return self._iter(payload, ctx)

    def _iter(self, payload: Payload, ctx: Context) -> Iterator[Signal]:
        retrieval = gather_retrieval(ctx)
        if not retrieval:
            return
        sentences = list(split_sentences(payload.text))
        eligible: list[tuple[int, str]] = []
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < self._min_sentence_tokens:
                continue
            eligible.append((i, sentence))
        if not eligible:
            return

        batch = [retrieval] + [s for _, s in eligible]
        vectors = self._embedder(batch)
        if len(vectors) != len(batch):
            raise RuntimeError(
                "embedder returned wrong number of vectors: "
                f"expected {len(batch)}, got {len(vectors)}"
            )
        retrieval_vec = vectors[0]
        offset = 0
        for (i, sentence), sent_vec in zip(eligible, vectors[1:]):
            sim = cosine_similarity(sent_vec, retrieval_vec)
            if sim >= self._threshold:
                continue
            start = payload.text.find(sentence, offset)
            end = start + len(sentence) if start >= 0 else None
            offset = end if end is not None else offset
            yield Signal(
                detector=self._name,
                category=OwaspCategory.LLM09_MISINFORMATION,
                severity=self._severity,
                confidence=1.0 - sim,
                message=(
                    f"Sentence cosine similarity {sim:.2f} below threshold "
                    f"{self._threshold:.2f}"
                ),
                span=(start, end) if start >= 0 and end is not None else None,
                matched_text=sentence[:160],
                metadata={
                    "similarity": sim,
                    "threshold": self._threshold,
                    "sentence_index": i,
                },
            )


# ---------------------------------------------------------------------------
# Sentence-transformer factory
# ---------------------------------------------------------------------------


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Known sentence-transformer models worth surfacing. These all return
#: vectors small enough (< 1024-d) to keep memory and CPU costs sane.
KNOWN_EMBEDDING_MODELS: tuple[str, ...] = (
    "sentence-transformers/all-MiniLM-L6-v2",       # 384-d, fast, default
    "sentence-transformers/all-mpnet-base-v2",      # 768-d, higher quality
    "BAAI/bge-small-en-v1.5",                       # 384-d
    "BAAI/bge-base-en-v1.5",                        # 768-d
    "intfloat/e5-small-v2",                         # 384-d
    "intfloat/e5-base-v2",                          # 768-d
)


def sentence_transformer_embedder(
    model: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
    normalize: bool = True,
) -> Embedder:
    """Build an :data:`Embedder` callable backed by sentence-transformers.

    Requires ``pip install soweak[embeddings]`` (which transitively pulls
    ``transformers`` and ``torch``).

    Parameters:
      model: HF model id. Defaults to ``all-MiniLM-L6-v2`` (384-d, fast).
      device: ``"cpu"`` (default), ``"cuda"``, ``"mps"``, etc.
      normalize: L2-normalise vectors before returning them. Cosine
        similarity on normalised vectors is the same as a dot product;
        enabling this is the recommended default.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError(
            "EmbeddingGroundingDetector with sentence-transformers requires "
            "`pip install soweak[embeddings]`."
        ) from e

    st_model = SentenceTransformer(model, device=device)

    def _embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = st_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in encoded]

    _embed.model_name = model  # type: ignore[attr-defined]
    return _embed


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "Embedder",
    "EmbeddingGroundingDetector",
    "KNOWN_EMBEDDING_MODELS",
    "cosine_similarity",
    "sentence_transformer_embedder",
]
