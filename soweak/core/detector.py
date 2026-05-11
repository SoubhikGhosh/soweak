"""Detector ABC and Signal dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

from soweak.core.types import Boundary, Context, OwaspCategory, Payload, Severity


@dataclass
class Signal:
    """A single observation produced by a Detector."""

    detector: str
    category: OwaspCategory
    severity: Severity
    confidence: float = 1.0
    message: str = ""
    span: tuple[int, int] | None = None
    matched_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Detector(ABC):
    """A signal producer.

    Subclasses implement :meth:`inspect` and declare which boundaries they
    apply to via :attr:`boundaries`. The framework runs them at the boundaries
    the policy attaches them to — :attr:`boundaries` is advisory documentation,
    not a hard constraint.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in signals and audit logs."""

    @property
    @abstractmethod
    def category(self) -> OwaspCategory:
        """Primary OWASP category this detector addresses."""

    @property
    def boundaries(self) -> tuple[Boundary, ...]:
        """Boundaries where this detector is meaningful. Default: input only."""
        return (Boundary.INPUT,)

    @abstractmethod
    def inspect(self, payload: Payload, ctx: Context) -> Iterable[Signal]:
        """Inspect a payload and yield zero or more signals."""

    async def ainspect(self, payload: Payload, ctx: Context) -> list[Signal]:
        """Async variant of :meth:`inspect`.

        Default implementation delegates to the sync ``inspect`` so existing
        detectors work in an async :class:`~soweak.Pipeline` without changes.
        Override when your detector performs real I/O (calls to a hosted
        classifier, vector store, external policy engine, etc.).
        """
        return list(self.inspect(payload, ctx))
