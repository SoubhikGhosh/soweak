"""Built-in enforcers: Block, Redact, LogOnly, Threshold, Transform.

These are the action layer. They take signals from one or more detectors and
return a :class:`Decision` telling the pipeline whether to allow, warn,
redact, transform, require approval, or block.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from soweak.core.detector import Signal
from soweak.core.enforcer import Action, Decision, Enforcer
from soweak.core.types import Context, Payload, Severity


def _max_severity(signals: list[Signal]) -> Severity | None:
    if not signals:
        return None
    return max(s.severity for s in signals)


class BlockEnforcer(Enforcer):
    """Block when any signal meets or exceeds ``min_severity``.

    Below the threshold, allow with a WARN if any signal fired, ALLOW
    otherwise.
    """

    def __init__(
        self,
        min_severity: Severity = Severity.HIGH,
        name: str = "block",
    ) -> None:
        self._min_severity = min_severity
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def decide(
        self, payload: Payload, signals: list[Signal], ctx: Context
    ) -> Decision:
        max_sev = _max_severity(signals)
        if max_sev is not None and max_sev >= self._min_severity:
            return Decision(
                action=Action.BLOCK,
                payload=payload,
                signals=list(signals),
                reason=(
                    f"max severity {max_sev.label} "
                    f">= {self._min_severity.label}"
                ),
            )
        if signals:
            return Decision(Action.WARN, payload, list(signals))
        return Decision.allow(payload)


class RedactEnforcer(Enforcer):
    """Replace matched spans in the payload text with a placeholder.

    Only signals with a ``span`` and severity ≥ ``min_severity`` are redacted.
    """

    def __init__(
        self,
        placeholder: str = "[REDACTED]",
        min_severity: Severity = Severity.LOW,
        name: str = "redact",
    ) -> None:
        self._placeholder = placeholder
        self._min_severity = min_severity
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def decide(
        self, payload: Payload, signals: list[Signal], ctx: Context
    ) -> Decision:
        eligible = [
            s
            for s in signals
            if s.span is not None and s.severity >= self._min_severity
        ]
        if not eligible:
            return Decision(
                Action.WARN if signals else Action.ALLOW,
                payload,
                list(signals),
            )
        # Replace spans back-to-front so earlier offsets stay valid.
        spans = sorted(
            ((s.span[0], s.span[1]) for s in eligible if s.span is not None),
            key=lambda x: x[0],
            reverse=True,
        )
        text = payload.text
        for start, end in spans:
            text = text[:start] + self._placeholder + text[end:]
        new_payload = replace(payload, text=text)
        return Decision(
            Action.REDACT,
            new_payload,
            list(signals),
            reason=f"redacted {len(spans)} span(s)",
        )


class LogOnlyEnforcer(Enforcer):
    """Never modifies the payload. WARN if any signal fired, ALLOW otherwise."""

    def __init__(self, name: str = "log-only") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def decide(
        self, payload: Payload, signals: list[Signal], ctx: Context
    ) -> Decision:
        action = Action.WARN if signals else Action.ALLOW
        return Decision(action, payload, list(signals))


class ThresholdEnforcer(Enforcer):
    """Score = Σ(severity_weight × confidence). Block above ``block_at``,
    warn above ``warn_at``, otherwise allow.

    Useful when you want graceful escalation rather than a single severity
    cutoff.
    """

    def __init__(
        self,
        block_at: float = 1.0,
        warn_at: float = 0.5,
        name: str = "threshold",
    ) -> None:
        if block_at < warn_at:
            raise ValueError("block_at must be >= warn_at")
        self._block_at = block_at
        self._warn_at = warn_at
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def decide(
        self, payload: Payload, signals: list[Signal], ctx: Context
    ) -> Decision:
        score = sum(s.severity.weight * s.confidence for s in signals)
        if score >= self._block_at:
            return Decision(
                Action.BLOCK,
                payload,
                list(signals),
                reason=f"score {score:.2f} >= block_at {self._block_at}",
                metadata={"score": score},
            )
        if score >= self._warn_at:
            return Decision(
                Action.WARN,
                payload,
                list(signals),
                reason=f"score {score:.2f} >= warn_at {self._warn_at}",
                metadata={"score": score},
            )
        return Decision(
            Action.ALLOW, payload, list(signals), metadata={"score": score}
        )


class TransformEnforcer(Enforcer):
    """Apply a caller-supplied function to the payload text."""

    def __init__(
        self,
        transform: Callable[[str], str],
        name: str = "transform",
    ) -> None:
        self._transform = transform
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def decide(
        self, payload: Payload, signals: list[Signal], ctx: Context
    ) -> Decision:
        new_text = self._transform(payload.text)
        new_payload = replace(payload, text=new_text)
        return Decision(
            Action.TRANSFORM,
            new_payload,
            list(signals),
            reason="payload transformed",
        )


__all__ = [
    "BlockEnforcer",
    "LogOnlyEnforcer",
    "RedactEnforcer",
    "ThresholdEnforcer",
    "TransformEnforcer",
]
