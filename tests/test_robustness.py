"""Robustness tests: ReDoS guard, i18n grounding, large-input safety.

Pattern packs are user-facing surface. A poorly written regex can produce
catastrophic backtracking on adversarial input. These tests guard against
that for every built-in pack and verify that the i18n-friendly grounding
tokenizer accepts non-Latin scripts.
"""

from __future__ import annotations

import re
import signal
from contextlib import contextmanager
from typing import Iterator

import pytest

from soweak import Boundary, Context, Payload
from soweak.detectors import (
    INPUT_DLP_PACK,
    OUTPUT_DLP_PACK,
    OUTPUT_HTML_PACK,
    OUTPUT_SHELL_PACK,
    OUTPUT_SQL_PACK,
    PROMPT_INJECTION_PACK,
    SYSTEM_PROMPT_EXTRACTION_PACK,
    PatternMatchDetector,
)
from soweak.detectors.patterns import PatternPack
from soweak.grounding import gather_retrieval, split_sentences, tokenize


ALL_PACKS: list[PatternPack] = [
    PROMPT_INJECTION_PACK,
    INPUT_DLP_PACK,
    OUTPUT_DLP_PACK,
    OUTPUT_HTML_PACK,
    OUTPUT_SQL_PACK,
    OUTPUT_SHELL_PACK,
    SYSTEM_PROMPT_EXTRACTION_PACK,
]


# ---------------------------------------------------------------------------
# Regex compile guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pack", ALL_PACKS, ids=lambda p: p.name)
def test_every_pattern_compiles(pack: PatternPack) -> None:
    """Construction of the PatternMatchDetector compiles every pattern."""
    PatternMatchDetector(pack)


# ---------------------------------------------------------------------------
# ReDoS guard
# ---------------------------------------------------------------------------


@contextmanager
def _timeout(seconds: float) -> Iterator[None]:
    """Linux/macOS-only SIGALRM-based timer."""

    def _handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"regex took longer than {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


REDOS_PROBES = [
    "a" * 5_000,
    "ab" * 5_000,
    "\n" * 5_000,
    "<" * 5_000 + "script" + ">" * 5_000,
    "/* " * 1_000 + " */",
    "Ignore " * 2_000,
    " " * 10_000,
    "0" * 10_000,
    "AKIA" + "X" * 10_000,
    "x" * 5_000 + "@" + "y" * 5_000 + ".com",
]


@pytest.mark.parametrize("pack", ALL_PACKS, ids=lambda p: p.name)
@pytest.mark.parametrize("probe", REDOS_PROBES, ids=lambda s: f"len{len(s)}")
def test_packs_resist_redos(pack: PatternPack, probe: str) -> None:
    """Each pack must process every adversarial probe in < 1.5 seconds."""
    det = PatternMatchDetector(pack)
    payload = Payload(Boundary.INPUT, text=probe)
    with _timeout(1.5):
        list(det.inspect(payload, Context()))


# ---------------------------------------------------------------------------
# i18n grounding tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_handles_latin() -> None:
    out = tokenize("The Eiffel Tower is in Paris.")
    assert "eiffel" in out
    assert "paris" in out


def test_tokenize_handles_cjk() -> None:
    """3+ codepoint CJK runs should tokenize, not be filtered out."""
    out = tokenize("人工知能のセキュリティ")  # "artificial intelligence security"
    assert out  # at minimum, some token survives


def test_tokenize_handles_cyrillic() -> None:
    out = tokenize("Большая языковая модель")  # "Large language model"
    assert "большая" in out or "языковая" in out or "модель" in out


def test_tokenize_handles_arabic() -> None:
    out = tokenize("نموذج اللغة الكبير")  # "the large language model"
    assert out  # some Arabic tokens survive


def test_tokenize_drops_punctuation() -> None:
    out = tokenize("hello, world! how-are-you.")
    assert "," not in out
    assert "!" not in out


def test_split_sentences_handles_unicode_terminators() -> None:
    sentences = split_sentences("こんにちは。元気ですか。")
    assert len(sentences) == 2


def test_split_sentences_handles_ascii() -> None:
    sentences = split_sentences("Hello! How are you? I am fine.")
    assert len(sentences) == 3


# ---------------------------------------------------------------------------
# gather_retrieval public API
# ---------------------------------------------------------------------------


def test_gather_retrieval_from_text_key() -> None:
    ctx = Context(metadata={"retrieved_text": "hello world"})
    assert gather_retrieval(ctx) == "hello world"


def test_gather_retrieval_from_documents_list() -> None:
    ctx = Context(metadata={"retrieved_documents": [{"text": "a"}, {"text": "b"}]})
    out = gather_retrieval(ctx)
    assert "a" in out and "b" in out


def test_gather_retrieval_empty_context() -> None:
    assert gather_retrieval(Context()) == ""


# ---------------------------------------------------------------------------
# Pack version validation
# ---------------------------------------------------------------------------


def test_pack_require_version_accepts_current() -> None:
    PROMPT_INJECTION_PACK.require_version("1.0")


def test_pack_require_version_rejects_future() -> None:
    with pytest.raises(ValueError, match="requires >= 99"):
        PROMPT_INJECTION_PACK.require_version("99.0")


def test_pack_require_version_validates_format() -> None:
    with pytest.raises(ValueError, match="invalid pack version"):
        PROMPT_INJECTION_PACK.require_version("not-a-version")
