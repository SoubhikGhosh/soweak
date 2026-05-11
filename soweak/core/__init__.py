"""Core framework primitives.

Public re-exports live on the top-level ``soweak`` package; this module is the
implementation home.
"""

from soweak.core.audit import (
    AuditEvent,
    AuditLog,
    InMemoryAuditLog,
    JsonLinesAuditLog,
)
from soweak.core.detector import Detector, Signal
from soweak.core.enforcer import Action, Decision, Enforcer
from soweak.core.pipeline import Pipeline
from soweak.core.policy import Policy, PolicyBuilder, Rule
from soweak.core.types import (
    Boundary,
    Context,
    OwaspCategory,
    Payload,
    Severity,
)

__all__ = [
    "Action",
    "AuditEvent",
    "AuditLog",
    "Boundary",
    "Context",
    "Decision",
    "Detector",
    "Enforcer",
    "InMemoryAuditLog",
    "JsonLinesAuditLog",
    "OwaspCategory",
    "Payload",
    "Pipeline",
    "Policy",
    "PolicyBuilder",
    "Rule",
    "Severity",
    "Signal",
]
