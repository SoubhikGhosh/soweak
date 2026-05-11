"""Policy, Rule, and PolicyBuilder.

A Policy is an ordered set of Rules, each binding a boundary to one or more
Detectors and exactly one Enforcer. PolicyBuilder provides a fluent API for
constructing policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from soweak.core.detector import Detector
from soweak.core.enforcer import Enforcer
from soweak.core.types import Boundary


@dataclass(frozen=True)
class Rule:
    """One enforcement rule at a boundary."""

    boundary: Boundary
    detectors: tuple[Detector, ...]
    enforcer: Enforcer
    name: str = ""


@dataclass(frozen=True)
class Policy:
    """An ordered set of rules. Immutable once built."""

    rules: tuple[Rule, ...] = field(default_factory=tuple)

    def for_boundary(self, boundary: Boundary) -> tuple[Rule, ...]:
        """Return rules attached to a given boundary, in order."""
        return tuple(r for r in self.rules if r.boundary == boundary)

    @classmethod
    def builder(cls) -> PolicyBuilder:
        return PolicyBuilder()


class _BoundaryClause:
    """Fluent intermediate returned by PolicyBuilder.on_*().

    Collects detectors via ``.detect(...)`` and finalises with ``.enforce(...)``,
    which commits the rule and returns control to the parent PolicyBuilder.
    """

    def __init__(self, parent: PolicyBuilder, boundary: Boundary, name: str) -> None:
        self._parent = parent
        self._boundary = boundary
        self._name = name
        self._detectors: list[Detector] = []

    def detect(self, *detectors: Detector) -> _BoundaryClause:
        """Add one or more detectors to this rule."""
        self._detectors.extend(detectors)
        return self

    def enforce(self, enforcer: Enforcer) -> PolicyBuilder:
        """Commit this rule with the given enforcer and return the builder."""
        self._parent._commit(
            Rule(
                boundary=self._boundary,
                detectors=tuple(self._detectors),
                enforcer=enforcer,
                name=self._name,
            )
        )
        return self._parent


class PolicyBuilder:
    """Fluent builder for Policy objects.

    Example::

        policy = (
            PolicyBuilder()
            .on_input("user-input")
                .detect(prompt_injection_detector(), input_dlp_detector())
                .enforce(BlockEnforcer(min_severity=Severity.HIGH))
            .on_output("model-output")
                .detect(CanaryDetector(tokens=["sk-prod-..."]))
                .enforce(RedactEnforcer())
            .build()
        )
    """

    def __init__(self) -> None:
        self._rules: list[Rule] = []

    def on_input(self, name: str = "input") -> _BoundaryClause:
        return _BoundaryClause(self, Boundary.INPUT, name)

    def on_retrieval(self, name: str = "retrieval") -> _BoundaryClause:
        return _BoundaryClause(self, Boundary.RETRIEVAL, name)

    def on_tool_call(self, name: str = "tool_call") -> _BoundaryClause:
        return _BoundaryClause(self, Boundary.TOOL_CALL, name)

    def on_output(self, name: str = "output") -> _BoundaryClause:
        return _BoundaryClause(self, Boundary.OUTPUT, name)

    def on_stream(self, name: str = "stream") -> _BoundaryClause:
        return _BoundaryClause(self, Boundary.STREAM, name)

    def _commit(self, rule: Rule) -> None:
        self._rules.append(rule)

    def build(self) -> Policy:
        return Policy(rules=tuple(self._rules))
