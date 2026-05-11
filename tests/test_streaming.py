"""LLM10 streaming repetition detector tests."""

from __future__ import annotations

from soweak import Boundary, Context, OwaspCategory, Payload, RepetitionDetector


def _signals(text: str, **kw):
    det = RepetitionDetector(**kw)
    return list(det.inspect(Payload(Boundary.OUTPUT, text=text), Context()))


def test_repetition_fires_on_simple_loop():
    text = "loop loop loop loop loop loop loop loop loop loop "
    sig = _signals(text)
    assert sig
    assert sig[0].category is OwaspCategory.LLM10_UNBOUNDED_CONSUMPTION
    assert sig[0].metadata["repeats"] >= 5


def test_repetition_ignores_short_output():
    assert _signals("hi") == []
    assert _signals("loop loop loop") == []  # only 3 repeats


def test_repetition_ignores_unique_long_text():
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "Bright vixens jump dozily for quick wits. "
        "Sphinx of black quartz judge my vow today. "
    )
    assert _signals(text) == []


def test_repetition_threshold_tunable():
    text = "abcde" * 3  # 3 repeats of 5-char unit
    assert _signals(text, min_repeats=3) != []
    assert _signals(text, min_repeats=10) == []


def test_repetition_only_one_signal_per_call():
    text = "abc" * 20 + "xyz" * 20
    sigs = _signals(text)
    assert len(sigs) == 1  # short-circuits after first hit


def test_repetition_handles_empty_text():
    assert _signals("") == []


def test_repetition_default_boundaries():
    det = RepetitionDetector()
    assert Boundary.OUTPUT in det.boundaries
    assert Boundary.STREAM in det.boundaries


def test_repetition_validates_min_repeats():
    import pytest
    with pytest.raises(ValueError):
        RepetitionDetector(min_repeats=1)
