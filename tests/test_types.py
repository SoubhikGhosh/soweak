"""Tests for the core type machinery."""

from __future__ import annotations

from soweak import Boundary, Context, OwaspCategory, Payload, Severity


def test_severity_ordering() -> None:
    assert Severity.INFO < Severity.LOW < Severity.MEDIUM < Severity.HIGH < Severity.CRITICAL
    assert Severity.CRITICAL >= Severity.HIGH
    assert max(Severity.LOW, Severity.HIGH) == Severity.HIGH


def test_severity_weights() -> None:
    assert Severity.INFO.weight < Severity.CRITICAL.weight
    assert Severity.CRITICAL.weight == 1.0


def test_severity_label() -> None:
    assert Severity.HIGH.label == "high"
    assert Severity.CRITICAL.label == "critical"


def test_owasp_category_values() -> None:
    assert OwaspCategory.LLM01_PROMPT_INJECTION.value == "LLM01"
    assert OwaspCategory.LLM10_UNBOUNDED_CONSUMPTION.value == "LLM10"


def test_payload_defaults() -> None:
    p = Payload(Boundary.INPUT, text="hi")
    assert p.boundary is Boundary.INPUT
    assert p.text == "hi"
    assert p.raw is None
    assert p.metadata == {}


def test_context_request_id_unique() -> None:
    a, b = Context(), Context()
    assert a.request_id != b.request_id
