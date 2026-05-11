"""Enforcer decision tables."""

from __future__ import annotations

import pytest

from soweak import (
    Action,
    BlockEnforcer,
    Boundary,
    Context,
    LogOnlyEnforcer,
    OwaspCategory,
    Payload,
    RedactEnforcer,
    Severity,
    Signal,
    ThresholdEnforcer,
    TransformEnforcer,
)


def _sig(severity: Severity, span: tuple[int, int] | None = None, confidence: float = 1.0) -> Signal:
    return Signal(
        detector="test",
        category=OwaspCategory.LLM01_PROMPT_INJECTION,
        severity=severity,
        confidence=confidence,
        message="test signal",
        span=span,
        matched_text="x" if span else None,
    )


@pytest.fixture
def payload() -> Payload:
    return Payload(Boundary.INPUT, text="hello world")


@pytest.fixture
def ctx() -> Context:
    return Context()


def test_block_enforcer_blocks_at_threshold(payload: Payload, ctx: Context) -> None:
    enf = BlockEnforcer(min_severity=Severity.HIGH)
    d = enf.decide(payload, [_sig(Severity.HIGH)], ctx)
    assert d.action is Action.BLOCK
    assert d.blocked


def test_block_enforcer_warns_below_threshold(payload: Payload, ctx: Context) -> None:
    enf = BlockEnforcer(min_severity=Severity.HIGH)
    d = enf.decide(payload, [_sig(Severity.LOW)], ctx)
    assert d.action is Action.WARN
    assert not d.blocked


def test_block_enforcer_allows_when_no_signals(payload: Payload, ctx: Context) -> None:
    enf = BlockEnforcer()
    d = enf.decide(payload, [], ctx)
    assert d.action is Action.ALLOW


def test_redact_enforcer_replaces_span(ctx: Context) -> None:
    p = Payload(Boundary.INPUT, text="ABCDE secret FGHIJ")
    enf = RedactEnforcer(placeholder="[X]", min_severity=Severity.LOW)
    signals = [_sig(Severity.HIGH, span=(6, 12))]  # "secret"
    d = enf.decide(p, signals, ctx)
    assert d.action is Action.REDACT
    assert d.payload.text == "ABCDE [X] FGHIJ"


def test_redact_enforcer_ignores_signals_below_min(ctx: Context) -> None:
    p = Payload(Boundary.INPUT, text="ABCDE secret FGHIJ")
    enf = RedactEnforcer(min_severity=Severity.CRITICAL)
    signals = [_sig(Severity.MEDIUM, span=(6, 12))]
    d = enf.decide(p, signals, ctx)
    assert d.action is Action.WARN
    assert d.payload.text == p.text  # unchanged


def test_redact_enforcer_handles_overlapping_spans_back_to_front(ctx: Context) -> None:
    p = Payload(Boundary.INPUT, text="aa bb cc dd")
    signals = [
        _sig(Severity.HIGH, span=(0, 2)),
        _sig(Severity.HIGH, span=(6, 8)),
    ]
    d = RedactEnforcer(placeholder="[R]").decide(p, signals, ctx)
    assert d.payload.text == "[R] bb [R] dd"


def test_log_only_enforcer(payload: Payload, ctx: Context) -> None:
    enf = LogOnlyEnforcer()
    assert enf.decide(payload, [], ctx).action is Action.ALLOW
    assert enf.decide(payload, [_sig(Severity.CRITICAL)], ctx).action is Action.WARN


def test_threshold_enforcer_block_warn_allow(payload: Payload, ctx: Context) -> None:
    enf = ThresholdEnforcer(block_at=1.0, warn_at=0.4)
    # CRITICAL * 1.0 = 1.0 -> block
    assert enf.decide(payload, [_sig(Severity.CRITICAL)], ctx).action is Action.BLOCK
    # MEDIUM * 1.0 = 0.5 -> warn
    assert enf.decide(payload, [_sig(Severity.MEDIUM)], ctx).action is Action.WARN
    # LOW * 1.0 = 0.3 -> allow
    assert enf.decide(payload, [_sig(Severity.LOW)], ctx).action is Action.ALLOW


def test_threshold_validates_args() -> None:
    with pytest.raises(ValueError):
        ThresholdEnforcer(block_at=0.1, warn_at=0.5)


def test_transform_enforcer_runs_callable(payload: Payload, ctx: Context) -> None:
    enf = TransformEnforcer(transform=str.upper)
    d = enf.decide(payload, [], ctx)
    assert d.action is Action.TRANSFORM
    assert d.payload.text == "HELLO WORLD"
