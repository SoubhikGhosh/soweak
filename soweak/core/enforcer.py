"""Enforcer ABC, Action, Decision."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from soweak.core.detector import Signal
from soweak.core.types import Context, Payload


class Action(str, Enum):
    """The action an Enforcer decided on.

    ``BLOCK`` short-circuits the pipeline; everything else is informational
    (``WARN``) or transforms the payload (``REDACT``, ``TRANSFORM``).
    """

    ALLOW = "allow"
    WARN = "warn"
    REDACT = "redact"
    TRANSFORM = "transform"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"


@dataclass
class Decision:
    """The result of evaluating one rule against a payload."""

    action: Action
    payload: Payload
    signals: list[Signal] = field(default_factory=list)
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # ----- factories -----

    @classmethod
    def allow(
        cls, payload: Payload, signals: list[Signal] | None = None
    ) -> Decision:
        return cls(Action.ALLOW, payload, list(signals or []))

    @classmethod
    def warn(
        cls,
        payload: Payload,
        signals: list[Signal] | None = None,
        reason: str = "",
    ) -> Decision:
        return cls(Action.WARN, payload, list(signals or []), reason=reason)

    @classmethod
    def block(
        cls,
        payload: Payload,
        signals: list[Signal] | None = None,
        reason: str = "",
    ) -> Decision:
        return cls(Action.BLOCK, payload, list(signals or []), reason=reason)

    @classmethod
    def redact(
        cls,
        payload: Payload,
        signals: list[Signal] | None = None,
        reason: str = "",
    ) -> Decision:
        return cls(Action.REDACT, payload, list(signals or []), reason=reason)

    @classmethod
    def transform(
        cls,
        payload: Payload,
        signals: list[Signal] | None = None,
        reason: str = "",
    ) -> Decision:
        return cls(Action.TRANSFORM, payload, list(signals or []), reason=reason)

    # ----- predicates -----

    @property
    def blocked(self) -> bool:
        return self.action == Action.BLOCK

    @property
    def allowed(self) -> bool:
        return self.action in (
            Action.ALLOW,
            Action.WARN,
            Action.REDACT,
            Action.TRANSFORM,
        )


class Enforcer(ABC):
    """An action taker. Given a payload and the signals raised against it,
    produces a Decision."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def decide(self, payload: Payload, signals: list[Signal], ctx: Context) -> Decision:
        ...

    async def adecide(
        self, payload: Payload, signals: list[Signal], ctx: Context
    ) -> Decision:
        """Async variant of :meth:`decide`.

        Default implementation delegates to the sync ``decide`` so existing
        enforcers work inside ``Pipeline.arun`` without changes. Override when
        your enforcer awaits external I/O (a human-approval RPC, a remote
        policy decision point, etc.).
        """
        return self.decide(payload, signals, ctx)
