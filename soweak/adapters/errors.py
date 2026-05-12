"""Adapter-shared exception type."""

from __future__ import annotations

from soweak.core.detector import Signal
from soweak.core.enforcer import Decision


class SecurityError(RuntimeError):
    """Raised by adapters when a Pipeline returns a BLOCK decision."""

    def __init__(self, decision: Decision) -> None:
        super().__init__(
            decision.reason or f"blocked at boundary {decision.payload.boundary.value}"
        )
        self.decision = decision

    @property
    def signals(self) -> list[Signal]:
        return self.decision.signals
