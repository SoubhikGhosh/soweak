"""Shared fixtures."""

from __future__ import annotations

import pytest

from soweak import (
    BlockEnforcer,
    InMemoryAuditLog,
    Pipeline,
    PolicyBuilder,
    RedactEnforcer,
    Severity,
)
from soweak.detectors import (
    CanaryDetector,
    input_dlp_detector,
    prompt_injection_detector,
    system_prompt_extraction_detector,
)


@pytest.fixture
def canaries() -> list[str]:
    return ["x7K2-PRODSEC-9F4E"]


@pytest.fixture
def audit_log() -> InMemoryAuditLog:
    return InMemoryAuditLog()


@pytest.fixture
def default_pipeline(canaries: list[str], audit_log: InMemoryAuditLog) -> Pipeline:
    policy = (
        PolicyBuilder()
        .on_input("prompt-injection")
        .detect(prompt_injection_detector(), system_prompt_extraction_detector())
        .enforce(BlockEnforcer(min_severity=Severity.HIGH))
        .on_input("input-dlp")
        .detect(input_dlp_detector())
        .enforce(RedactEnforcer(min_severity=Severity.HIGH))
        .on_output("canary-leak")
        .detect(CanaryDetector(tokens=canaries))
        .enforce(BlockEnforcer(min_severity=Severity.CRITICAL))
        .build()
    )
    return Pipeline(policy, audit=audit_log)
