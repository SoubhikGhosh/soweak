"""PolicyBuilder and Policy."""

from __future__ import annotations

import pytest

from soweak import (
    BlockEnforcer,
    Boundary,
    LogOnlyEnforcer,
    Policy,
    PolicyBuilder,
    Severity,
)
from soweak.detectors import input_dlp_detector, prompt_injection_detector


def test_builder_attaches_rules_in_order() -> None:
    p = (
        PolicyBuilder()
        .on_input("a")
        .detect(prompt_injection_detector())
        .enforce(BlockEnforcer())
        .on_input("b")
        .detect(input_dlp_detector())
        .enforce(LogOnlyEnforcer())
        .build()
    )
    assert isinstance(p, Policy)
    rules = p.for_boundary(Boundary.INPUT)
    assert [r.name for r in rules] == ["a", "b"]
    assert isinstance(rules[0].enforcer, BlockEnforcer)
    assert isinstance(rules[1].enforcer, LogOnlyEnforcer)


def test_builder_separates_boundaries() -> None:
    p = (
        PolicyBuilder()
        .on_input("in")
        .detect(prompt_injection_detector())
        .enforce(BlockEnforcer())
        .on_output("out")
        .detect(prompt_injection_detector())
        .enforce(LogOnlyEnforcer())
        .build()
    )
    assert len(p.for_boundary(Boundary.INPUT)) == 1
    assert len(p.for_boundary(Boundary.OUTPUT)) == 1
    assert p.for_boundary(Boundary.RETRIEVAL) == ()


def test_policy_is_immutable() -> None:
    p = PolicyBuilder().build()
    with pytest.raises(Exception):
        p.rules = ()  # type: ignore[misc]


def test_builder_can_be_empty() -> None:
    p = PolicyBuilder().build()
    assert p.rules == ()


def test_policy_class_builder_alias() -> None:
    assert isinstance(Policy.builder(), PolicyBuilder)
