"""LLM09 — Misinformation: grounding and citation checks.

These are deliberately heuristic. Real misinformation defense needs a
fact-checking pipeline; what we ship is enough to flag obviously
ungrounded output for follow-up review.

* :class:`CitationRequiredDetector` — signals when long output makes
  claims without any citation marker.
* :class:`GroundingDetector` — computes lexical overlap between output
  sentences and the retrieval context the request used; flags sentences
  with low overlap as likely-ungrounded.

The grounding detector reads the retrieval context from
``ctx.metadata["retrieved_text"]`` (a string) or
``ctx.metadata["retrieved_documents"]`` (a list of doc-shaped values).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator

from soweak.core.detector import Detector, Signal
from soweak.core.types import Boundary, Context, OwaspCategory, Payload, Severity


#: Default citation patterns recognised: ``[ref]``, ``[1]``, ``(1)``, ``[doc-id]``.
DEFAULT_CITATION_REGEX = r"\[[\w_:-]+\]|\(\d{1,3}\)"

#: Well-known keys on ``Context.metadata`` for grounding context.
RETRIEVED_TEXT_KEY = "retrieved_text"
RETRIEVED_DOCS_KEY = "retrieved_documents"


class CitationRequiredDetector(Detector):
    """Flag output longer than ``min_chars`` that contains no citation marker."""

    def __init__(
        self,
        min_chars: int = 200,
        citation_regex: str = DEFAULT_CITATION_REGEX,
        severity: Severity = Severity.MEDIUM,
        name: str = "citation-required",
    ) -> None:
        if min_chars <= 0:
            raise ValueError("min_chars must be positive")
        self._min_chars = min_chars
        self._regex = re.compile(citation_regex)
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
        text = payload.text
        if len(text) < self._min_chars:
            return
        if self._regex.search(text):
            return
        yield Signal(
            detector=self._name,
            category=OwaspCategory.LLM09_MISINFORMATION,
            severity=self._severity,
            confidence=0.7,
            message=(
                f"Output is {len(text)} chars but contains no citation marker "
                f"matching {self._regex.pattern!r}"
            ),
        )


# Unicode-aware: match runs of "word" characters (letters in any script,
# digits, underscores) of length ≥3. The threshold of 3 keeps short noise
# words ("the", "and") out while not over-filtering non-Latin scripts
# whose words are often 2–3 codepoints.
_TOKEN_RE = re.compile(r"\b\w{3,}\b", re.UNICODE)
# Sentence terminators: ASCII ., !, ? plus common Unicode end punctuation
# (CJK 。, fullwidth ! ?, Arabic ؟, Urdu ۔). CJK scripts typically don't
# separate sentences with whitespace, so we match each sentence as a span
# of non-terminator characters followed by an optional terminator.
_TERMINATOR_CLASS = r"[.!?。!?؟۔]"
_SENTENCE_RE = re.compile(rf"[^.!?。!?؟۔]+{_TERMINATOR_CLASS}?", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "this",
        "that",
        "with",
        "from",
        "they",
        "have",
        "been",
        "were",
        "would",
        "could",
        "should",
        "what",
        "which",
        "their",
        "there",
        "these",
        "those",
        "about",
        "into",
        "than",
        "then",
        "when",
        "where",
        "while",
        "your",
        "will",
        "also",
    }
)


def tokenize(text: str) -> set[str]:
    """Return the set of grounding-relevant tokens in ``text``.

    Unicode-aware: matches words in any script (Latin, CJK, Arabic, etc.)
    of length ≥3 codepoints, lowercased, with common English stopwords
    removed. Public so :mod:`soweak.embeddings` and user code can build on
    the same normalisation soweak uses internally.
    """
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences on ASCII and Unicode terminators.

    Handles English (whitespace-separated sentences ending in . / ! / ?),
    CJK (no whitespace, terminators 。 ! ?), Arabic / Urdu (؟ ۔). Empty
    or whitespace-only fragments are dropped.
    """
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


def gather_retrieval(ctx: Context) -> str:
    """Pull retrieval context out of ``ctx.metadata``.

    Reads ``retrieved_text`` (a single string) first, then
    ``retrieved_documents`` (a list of doc-shaped values; supports plain
    strings, dicts with ``text`` / ``page_content`` / ``content`` /
    ``body`` keys, and objects exposing ``page_content`` / ``text``
    attributes — the same conventions used by :mod:`soweak.rag`).
    """
    direct = ctx.metadata.get(RETRIEVED_TEXT_KEY)
    if isinstance(direct, str) and direct:
        return direct
    docs = ctx.metadata.get(RETRIEVED_DOCS_KEY)
    if isinstance(docs, list):
        parts: list[str] = []
        for d in docs:
            if isinstance(d, str):
                parts.append(d)
            elif isinstance(d, dict):
                for k in ("text", "page_content", "content", "body"):
                    v = d.get(k)
                    if isinstance(v, str):
                        parts.append(v)
                        break
            else:
                v = getattr(d, "page_content", None) or getattr(d, "text", None)
                if isinstance(v, str):
                    parts.append(v)
        return "\n\n".join(parts)
    return ""


# Back-compat private aliases — internal callers used these. Will stay
# pointed at the public functions for the v3.x line.
_tokenize = tokenize
_split_sentences = split_sentences
_gather_retrieval = gather_retrieval


class GroundingDetector(Detector):
    """Flag output sentences with low lexical overlap against the retrieval
    context for the request.

    Heuristic: tokenise to ASCII words ≥4 chars, drop stopwords, compute the
    intersection-over-output-sentence ratio against retrieval tokens. Below
    ``min_overlap`` triggers a signal. Sentences shorter than
    ``min_sentence_tokens`` are skipped (too short to ground reliably).

    This is **not** a fact-checker. It cannot detect plausible-sounding
    fabrication that happens to share vocabulary with the source. Treat
    signals as "worth a human look", not "definitely false".
    """

    def __init__(
        self,
        min_overlap: float = 0.3,
        min_sentence_tokens: int = 4,
        retrieval_keys: tuple[str, ...] = (RETRIEVED_TEXT_KEY, RETRIEVED_DOCS_KEY),
        severity: Severity = Severity.LOW,
        name: str = "grounding",
    ) -> None:
        if not 0.0 < min_overlap <= 1.0:
            raise ValueError("min_overlap must be in (0, 1]")
        self._min_overlap = min_overlap
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
        retrieval = _gather_retrieval(ctx)
        if not retrieval:
            return  # nothing to ground against
        retrieval_tokens = _tokenize(retrieval)
        if not retrieval_tokens:
            return
        offset = 0
        for sentence in _split_sentences(payload.text):
            start = payload.text.find(sentence, offset)
            end = start + len(sentence) if start >= 0 else None
            offset = end if end is not None else offset
            tokens = _tokenize(sentence)
            if len(tokens) < self._min_sentence_tokens:
                continue
            overlap = len(tokens & retrieval_tokens) / len(tokens)
            if overlap < self._min_overlap:
                yield Signal(
                    detector=self._name,
                    category=OwaspCategory.LLM09_MISINFORMATION,
                    severity=self._severity,
                    confidence=0.6,
                    message=(
                        f"Sentence has {overlap:.0%} lexical overlap with retrieval context "
                        f"(threshold {self._min_overlap:.0%})"
                    ),
                    span=(start, end) if start >= 0 and end is not None else None,
                    matched_text=sentence[:160],
                    metadata={
                        "overlap": overlap,
                        "min_overlap": self._min_overlap,
                        "sentence_tokens": len(tokens),
                    },
                )


__all__ = [
    "CitationRequiredDetector",
    "DEFAULT_CITATION_REGEX",
    "GroundingDetector",
    "RETRIEVED_DOCS_KEY",
    "RETRIEVED_TEXT_KEY",
    "gather_retrieval",
    "split_sentences",
    "tokenize",
]
