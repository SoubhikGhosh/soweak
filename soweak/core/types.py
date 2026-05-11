"""Core data types: boundaries, severity, OWASP category, Payload, Context."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import Any


class Boundary(str, Enum):
    """The point in an LLM pipeline where a payload is being inspected."""

    INPUT = "input"
    RETRIEVAL = "retrieval"
    TOOL_CALL = "tool_call"
    OUTPUT = "output"
    STREAM = "stream"


@total_ordering
class Severity(Enum):
    """Signal severity. Ordered: INFO < LOW < MEDIUM < HIGH < CRITICAL."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value < other.value

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def weight(self) -> float:
        return _SEVERITY_WEIGHT[self]


_SEVERITY_WEIGHT: dict[Severity, float] = {
    Severity.INFO: 0.1,
    Severity.LOW: 0.3,
    Severity.MEDIUM: 0.5,
    Severity.HIGH: 0.8,
    Severity.CRITICAL: 1.0,
}


class OwaspCategory(str, Enum):
    """OWASP Top 10 for LLM Applications (2025)."""

    LLM01_PROMPT_INJECTION = "LLM01"
    LLM02_SENSITIVE_INFO = "LLM02"
    LLM03_SUPPLY_CHAIN = "LLM03"
    LLM04_DATA_POISONING = "LLM04"
    LLM05_OUTPUT_HANDLING = "LLM05"
    LLM06_EXCESSIVE_AGENCY = "LLM06"
    LLM07_SYSTEM_PROMPT_LEAKAGE = "LLM07"
    LLM08_VECTOR_EMBEDDING = "LLM08"
    LLM09_MISINFORMATION = "LLM09"
    LLM10_UNBOUNDED_CONSUMPTION = "LLM10"


@dataclass
class Payload:
    """A piece of content flowing through a boundary.

    ``text`` is the canonical text the detectors inspect. ``raw`` carries the
    original object (e.g., a tool-call dict, a list of retrieved docs) so
    enforcers can rebuild the structured form when needed.
    """

    boundary: Boundary
    text: str = ""
    raw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """Request-scoped context that travels with every Payload through a pipeline."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str | None = None
    tenant_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, **kwargs: Any) -> Context:
        return cls(**kwargs)
