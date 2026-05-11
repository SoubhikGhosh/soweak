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


def test_toxicity_factory_raises_without_extras():
    import sys

    from soweak import ml

    if "transformers" in sys.modules:
        pytest.skip("transformers is installed; this test exercises the missing-extras path")
    with pytest.raises(ImportError, match=r"soweak\[ml\]"):
        ml.transformers_toxicity_classifier()


# ---------------- known model registries ----------------


def test_known_injection_models_have_default():
    from soweak.ml import DEFAULT_HF_MODEL, KNOWN_INJECTION_MODELS

    assert DEFAULT_HF_MODEL in KNOWN_INJECTION_MODELS


def test_known_toxicity_models_have_default():
    from soweak.ml import DEFAULT_TOXICITY_MODEL, KNOWN_TOXICITY_MODELS

    assert DEFAULT_TOXICITY_MODEL in KNOWN_TOXICITY_MODELS


def test_known_model_entries_have_required_keys():
    from soweak.ml import KNOWN_INJECTION_MODELS, KNOWN_TOXICITY_MODELS

    for registry in (KNOWN_INJECTION_MODELS, KNOWN_TOXICITY_MODELS):
        for model, cfg in registry.items():
            assert "positive_label_index" in cfg, model
            assert "max_length" in cfg, model
            assert "description" in cfg, model


# ---------------- llm_judge_classifier ----------------


def test_llm_judge_extracts_score_from_response():
    from soweak.ml import llm_judge_classifier

    judge_responses = iter(["0.92", "0.05", "I think 0.74."])

    def fake_judge(prompt: str) -> str:
        return next(judge_responses)

    cls = llm_judge_classifier(fake_judge)
    assert cls("first") == pytest.approx(0.92)
    assert cls("second") == pytest.approx(0.05)
    assert cls("third") == pytest.approx(0.74)


def test_llm_judge_clamps_to_unit_interval():
    """Defensive parsing: responses outside [0,1] should be clamped."""
    from soweak.ml import llm_judge_classifier

    def echo(prompt: str) -> str:
        return "1.0"

    assert llm_judge_classifier(echo)("x") == 1.0


def test_llm_judge_default_to_zero_when_no_number_in_response():
    from soweak.ml import llm_judge_classifier

    def vague(prompt: str) -> str:
        return "I'm not sure but it seems okay."

    assert llm_judge_classifier(vague)("x") == 0.0


def test_llm_judge_empty_text_short_circuits():
    """Empty input is not even sent to the judge."""
    from soweak.ml import llm_judge_classifier

    calls: list[str] = []

    def judge(prompt: str) -> str:
        calls.append(prompt)
        return "0.99"

    cls = llm_judge_classifier(judge)
    assert cls("") == 0.0
    assert calls == []


def test_llm_judge_requires_text_placeholder():
    from soweak.ml import llm_judge_classifier

    with pytest.raises(ValueError, match=r"\{text\}"):
        llm_judge_classifier(lambda p: "0.5", prompt_template="no placeholder")


def test_llm_judge_custom_template_inserts_text():
    from soweak.ml import llm_judge_classifier

    received: list[str] = []

    def judge(prompt: str) -> str:
        received.append(prompt)
        return "0.4"

    cls = llm_judge_classifier(judge, prompt_template="rate this: {text} please")
    cls("HELLO")
    assert received == ["rate this: HELLO please"]


def test_llm_judge_custom_score_parser():
    from soweak.ml import llm_judge_classifier

    def judge(prompt: str) -> str:
        return "verdict: HIGH risk"

    def parse(resp: str) -> float:
        return 0.9 if "HIGH" in resp else 0.1

    cls = llm_judge_classifier(judge, score_parser=parse)
    assert cls("x") == pytest.approx(0.9)


def test_llm_judge_integrates_with_ml_classifier_detector():
    from soweak import (
        BlockEnforcer,
        Pipeline,
        PolicyBuilder,
        Severity,
    )
    from soweak.ml import llm_judge_classifier

    pipeline = Pipeline(
        PolicyBuilder()
        .on_input()
        .detect(
            MLClassifierDetector(
                classifier=llm_judge_classifier(lambda p: "0.95"),
                threshold=0.7,
            )
        )
        .enforce(BlockEnforcer(min_severity=Severity.HIGH))
        .build()
    )
    assert pipeline.check_input("anything").blocked
