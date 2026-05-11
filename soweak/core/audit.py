"""Audit log: AuditEvent + sinks (in-memory, JSON-lines file)."""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from soweak.core.detector import Signal
from soweak.core.enforcer import Decision
from soweak.core.types import Boundary


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class AuditEvent:
    """A single audit record covering one pipeline invocation at one boundary."""

    request_id: str
    boundary: Boundary
    signals: list[Signal]
    decision: Decision
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "boundary": self.boundary.value,
            "signals": [_signal_to_dict(s) for s in self.signals],
            "decision": {
                "action": self.decision.action.value,
                "reason": self.decision.reason,
                "metadata": self.decision.metadata,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)


def _signal_to_dict(s: Signal) -> dict[str, Any]:
    d = asdict(s)
    d["category"] = s.category.value
    d["severity"] = s.severity.label
    return d


class AuditLog(ABC):
    """Sink that receives every pipeline decision."""

    @abstractmethod
    def record(self, event: AuditEvent) -> None:
        ...


class InMemoryAuditLog(AuditLog):
    """Keeps events in a list. Thread-safe. Useful for tests and local dev."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._lock = threading.Lock()

    def record(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> list[AuditEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


class JsonLinesAuditLog(AuditLog):
    """Appends one JSON object per line to a file. Thread-safe."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, event: AuditEvent) -> None:
        line = event.to_json()
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
