"""MLClassifierDetector tests using fake classifiers (no transformers dep)."""

from __future__ import annotations

import pytest

from soweak import (
    Action,
    BlockEnforcer,
    Boundary,
    Context,
    MLClassifierDetector,
    OwaspCategory,
    Payload,
    Pipeline,
    PolicyBuilder,
    Severity,
)


def _high_classifier(text: str) -> float:
    """Toy classifier: 'attack' anywhere => probability 0.95."""
    return 0.95 if "attack" in text.lower() else 0.05


def test_ml_classifier_fires_above_threshold():
    det = MLClassifierDetector(classifier=_high_classifier, threshold=0.5)
    sigs = list(det.inspect(Payload(Boundary.INPUT, text="this is an attack"), Context()))
    assert len(sigs) == 1
    assert sigs[0].confidence == pytest.approx(0.95)
    assert sigs[0].category is OwaspCategory.LLM01_PROMPT_INJECTION


def test_ml_classifier_silent_below_threshold():
    det = MLClassifierDetector(classifier=_high_classifier, threshold=0.99)
    sigs = list(det.inspect(Payload(Boundary.INPUT, text="this is an attack"), Context()))
    assert sigs == []


def test_ml_classifier_skips_empty_text():
    called = False

    def cls(text: str) -> float:
        nonlocal called
        called = True
        return 0.99

    det = MLClassifierDetector(classifier=cls, threshold=0.5)
    sigs = list(det.inspect(Payload(Boundary.INPUT, text=""), Context()))
    assert sigs == []
    assert called is False


def test_ml_classifier_validates_threshold():
    with pytest.raises(ValueError):
        MLClassifierDetector(classifier=_high_classifier, threshold=-0.1)
    with pytest.raises(ValueError):
        MLClassifierDetector(classifier=_high_classifier, threshold=1.5)


def test_ml_classifier_custom_category_and_severity():
    det = MLClassifierDetector(
        classifier=_high_classifier,
        threshold=0.5,
        category=OwaspCategory.LLM05_OUTPUT_HANDLING,
        severity=Severity.CRITICAL,
    )
    sig = next(iter(det.inspect(Payload(Boundary.OUTPUT, text="attack"), Context())))
    assert sig.category is OwaspCategory.LLM05_OUTPUT_HANDLING
    assert sig.severity is Severity.CRITICAL


def test_ml_classifier_pipeline_integration():
    pipeline = Pipeline(
        PolicyBuilder()
        .on_input()
        .detect(MLClassifierDetector(classifier=_high_classifier, threshold=0.5))
        .enforce(BlockEnforcer(min_severity=Severity.HIGH))
        .build()
    )
    assert pipeline.check_input("attack me").blocked
    assert pipeline.check_input("hello world").action is Action.ALLOW


def test_ml_classifier_with_stateful_callable():
    """The classifier protocol is just Callable; closures and instances both work."""
    calls: list[str] = []

    def cls(text: str) -> float:
        calls.append(text)
        return 0.9

    det = MLClassifierDetector(classifier=cls, threshold=0.5)
    list(det.inspect(Payload(Boundary.INPUT, text="hi"), Context()))
    list(det.inspect(Payload(Boundary.INPUT, text="there"), Context()))
    assert calls == ["hi", "there"]


def test_transformers_factory_raises_without_extras():
    """Importing the factory works, but calling it without transformers fails clearly."""
    import sys

    from soweak import ml

    if "transformers" in sys.modules:
        pytest.skip("transformers is installed; this test exercises the missing-extras path")
    with pytest.raises(ImportError, match=r"soweak\[ml\]"):
        ml.transformers_classifier()
