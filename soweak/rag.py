"""LLM08 — Vector & Embedding Weaknesses: retriever middleware.

These detectors run at :attr:`Boundary.RETRIEVAL` against ``payload.raw``
(the structured list of retrieved documents). The pipeline's
``check_retrieval(documents, ctx)`` ergonomic helper sets up the payload
for you.

Supported document shapes:

* ``str`` — text only, no metadata
* ``dict`` — looks at ``"text"`` / ``"page_content"`` / ``"content"`` /
  ``"body"`` for text, and ``"metadata"`` (a sub-dict) or the top level for
  tenant / source / score keys.
* objects with ``.page_content`` and ``.metadata`` attributes (LangChain).
"""

from __future__ import annotations

import statistics
from typing import Any, Iterable, Iterator

from soweak.core.detector import Detector, Signal
from soweak.core.types import Boundary, Context, OwaspCategory, Payload, Severity
from soweak.detectors.pattern_match import PatternMatchDetector
from soweak.detectors.patterns import PROMPT_INJECTION_PACK


# ---------------------------------------------------------------------------
# document helpers
# ---------------------------------------------------------------------------


def _doc_text(doc: Any) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for k in ("text", "page_content", "content", "body"):
            v = doc.get(k)
            if isinstance(v, str):
                return v
    for attr in ("page_content", "text"):
        v = getattr(doc, attr, None)
        if isinstance(v, str):
            return v
    return ""


def _doc_metadata(doc: Any) -> dict[str, Any]:
    if isinstance(doc, dict):
        meta = doc.get("metadata")
        if isinstance(meta, dict):
            return meta
        # Treat top-level keys as metadata when no nested metadata exists.
        return {k: v for k, v in doc.items() if k not in ("text", "page_content", "content", "body")}
    meta = getattr(doc, "metadata", None)
    if isinstance(meta, dict):
        return meta
    return {}


def _doc_score(doc: Any) -> float | None:
    meta = _doc_metadata(doc)
    for k in ("score", "relevance_score", "similarity"):
        v = meta.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _iter_docs(payload: Payload) -> Iterator[tuple[int, Any]]:
    raw = payload.raw
    if isinstance(raw, list):
        for i, d in enumerate(raw):
            yield i, d


# ---------------------------------------------------------------------------
# IndirectInjectionDetector
# ---------------------------------------------------------------------------


class IndirectInjectionDetector(Detector):
    """Run prompt-injection patterns against retrieved document text.

    Indirect (a.k.a. 2nd-order) prompt injection is the attack vector where
    a malicious payload lives inside a document the retriever returns to
    the model. Use this as the LLM01 defense at :attr:`Boundary.RETRIEVAL`.
    """

    def __init__(
        self,
        name: str = "indirect-injection",
        boundaries: tuple[Boundary, ...] = (Boundary.RETRIEVAL,),
    ) -> None:
        self._inner = PatternMatchDetector(PROMPT_INJECTION_PACK)
        self._name = name
        self._boundaries = boundaries

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> OwaspCategory:
        return OwaspCategory.LLM01_PROMPT_INJECTION

    @property
    def boundaries(self) -> tuple[Boundary, ...]:
        return self._boundaries

    def inspect(self, payload: Payload, ctx: Context) -> Iterable[Signal]:
        return self._iter(payload, ctx)

    def _iter(self, payload: Payload, ctx: Context) -> Iterator[Signal]:
        for i, doc in _iter_docs(payload):
            text = _doc_text(doc)
            if not text:
                continue
            doc_payload = Payload(Boundary.RETRIEVAL, text=text, raw=doc)
            for sig in self._inner.inspect(doc_payload, ctx):
                # Re-emit under our name with the doc index tagged in metadata.
                yield Signal(
                    detector=self._name,
                    category=sig.category,
                    severity=sig.severity,
                    confidence=sig.confidence,
                    message=f"document[{i}]: {sig.message}",
                    span=sig.span,
                    matched_text=sig.matched_text,
                    metadata={**sig.metadata, "doc_index": i},
                )


# ---------------------------------------------------------------------------
# TenantIsolationDetector
# ---------------------------------------------------------------------------


class TenantIsolationDetector(Detector):
    """Flag retrieved documents whose tenant key doesn't match ``ctx.tenant_id``.

    Cross-tenant data leakage via shared vector stores is one of the most
    common LLM08 failure modes. Configure your retriever to embed a
    ``tenant_id`` (or other key) in document metadata; this detector
    verifies the retrieval honoured it.
    """

    def __init__(
        self,
        tenant_key: str = "tenant_id",
        name: str = "tenant-isolation",
    ) -> None:
        self._tenant_key = tenant_key
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> OwaspCategory:
        return OwaspCategory.LLM08_VECTOR_EMBEDDING

    @property
    def boundaries(self) -> tuple[Boundary, ...]:
        return (Boundary.RETRIEVAL,)

    def inspect(self, payload: Payload, ctx: Context) -> Iterable[Signal]:
        return self._iter(payload, ctx)

    def _iter(self, payload: Payload, ctx: Context) -> Iterator[Signal]:
        request_tenant = ctx.tenant_id
        if not request_tenant:
            return  # nothing to enforce
        for i, doc in _iter_docs(payload):
            meta = _doc_metadata(doc)
            doc_tenant = meta.get(self._tenant_key)
            if doc_tenant is None:
                yield Signal(
                    detector=self._name,
                    category=OwaspCategory.LLM08_VECTOR_EMBEDDING,
                    severity=Severity.HIGH,
                    confidence=0.95,
                    message=(
                        f"document[{i}] missing {self._tenant_key!r}; "
                        f"cannot verify tenant {request_tenant!r}"
                    ),
                    metadata={"doc_index": i, "request_tenant": request_tenant},
                )
                continue
            if doc_tenant != request_tenant:
                yield Signal(
                    detector=self._name,
                    category=OwaspCategory.LLM08_VECTOR_EMBEDDING,
                    severity=Severity.CRITICAL,
                    confidence=1.0,
                    message=(
                        f"document[{i}] tenant={doc_tenant!r} but "
                        f"request tenant={request_tenant!r}"
                    ),
                    metadata={
                        "doc_index": i,
                        "doc_tenant": doc_tenant,
                        "request_tenant": request_tenant,
                    },
                )


# ---------------------------------------------------------------------------
# ProvenanceDetector
# ---------------------------------------------------------------------------


class ProvenanceDetector(Detector):
    """Flag retrieved documents that lack any provenance field.

    Without provenance you cannot cite, deduplicate, or revoke a document.
    Configure ``required_keys`` to your retriever's convention — at least
    one must be present per document.
    """

    def __init__(
        self,
        required_keys: tuple[str, ...] = ("source", "url", "uri", "doc_id"),
        name: str = "provenance",
    ) -> None:
        if not required_keys:
            raise ValueError("required_keys must be non-empty")
        self._required = required_keys
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> OwaspCategory:
        return OwaspCategory.LLM08_VECTOR_EMBEDDING

    @property
    def boundaries(self) -> tuple[Boundary, ...]:
        return (Boundary.RETRIEVAL,)

    def inspect(self, payload: Payload, ctx: Context) -> Iterable[Signal]:
        return self._iter(payload)

    def _iter(self, payload: Payload) -> Iterator[Signal]:
        for i, doc in _iter_docs(payload):
            meta = _doc_metadata(doc)
            if not any(meta.get(k) for k in self._required):
                yield Signal(
                    detector=self._name,
                    category=OwaspCategory.LLM08_VECTOR_EMBEDDING,
                    severity=Severity.MEDIUM,
                    confidence=0.95,
                    message=(
                        f"document[{i}] lacks provenance "
                        f"(none of {list(self._required)} set)"
                    ),
                    metadata={"doc_index": i, "required_keys": list(self._required)},
                )


# ---------------------------------------------------------------------------
# RetrievalAnomalyDetector
# ---------------------------------------------------------------------------


class RetrievalAnomalyDetector(Detector):
    """Flag retrieval score outliers — a document scoring far below the rest
    of the batch often signals a poisoned or irrelevant injection.

    Uses simple z-score-like deviation from the median. Requires docs to
    carry a numeric score under ``metadata.score`` (or ``relevance_score``
    / ``similarity``).
    """

    def __init__(
        self,
        max_deviation: float = 3.0,
        name: str = "retrieval-anomaly",
    ) -> None:
        if max_deviation <= 0:
            raise ValueError("max_deviation must be positive")
        self._max_deviation = max_deviation
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> OwaspCategory:
        return OwaspCategory.LLM08_VECTOR_EMBEDDING

    @property
    def boundaries(self) -> tuple[Boundary, ...]:
        return (Boundary.RETRIEVAL,)

    def inspect(self, payload: Payload, ctx: Context) -> Iterable[Signal]:
        return self._iter(payload)

    def _iter(self, payload: Payload) -> Iterator[Signal]:
        scored: list[tuple[int, float]] = []
        for i, doc in _iter_docs(payload):
            score = _doc_score(doc)
            if score is not None:
                scored.append((i, score))
        if len(scored) < 3:
            return  # need at least 3 to compute deviation
        scores = [s for _, s in scored]
        median = statistics.median(scores)
        # Mean absolute deviation as a robust spread.
        mad = statistics.median(abs(s - median) for s in scores) or 1e-9
        for i, s in scored:
            deviation = abs(s - median) / mad
            if deviation > self._max_deviation:
                yield Signal(
                    detector=self._name,
                    category=OwaspCategory.LLM08_VECTOR_EMBEDDING,
                    severity=Severity.MEDIUM,
                    confidence=0.7,
                    message=(
                        f"document[{i}] score={s:.3f} deviates {deviation:.1f}× "
                        f"from median {median:.3f}"
                    ),
                    metadata={
                        "doc_index": i,
                        "score": s,
                        "median": median,
                        "deviation": deviation,
                    },
                )


__all__ = [
    "IndirectInjectionDetector",
    "ProvenanceDetector",
    "RetrievalAnomalyDetector",
    "TenantIsolationDetector",
]
