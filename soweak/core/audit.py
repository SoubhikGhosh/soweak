"""Audit log: AuditEvent + sinks (in-memory, JSON-lines file).

Every :class:`AuditLog` implementation exposes synchronous
:meth:`AuditLog.record` and async :meth:`AuditLog.arecord`. The async
default delegates to the sync impl, so existing sync sinks work in async
pipelines without changes. Override ``arecord`` when you want non-blocking
I/O (network sinks, async DB clients, etc.).

Sinks that hold OS resources implement context-manager protocol; call
:meth:`AuditLog.close` (or use ``with`` syntax) to release them.
"""

from __future__ import annotations

import asyncio
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import IO, Any

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
    """Sink that receives every pipeline decision.

    Synchronous backends implement :meth:`record`. Async-aware backends
    can override :meth:`arecord` to perform non-blocking writes; the
    default ``arecord`` runs ``record`` in a worker thread, keeping
    soweak's :meth:`~soweak.Pipeline.arun` from blocking the event loop on
    slow sinks.
    """

    @abstractmethod
    def record(self, event: AuditEvent) -> None:
        ...

    async def arecord(self, event: AuditEvent) -> None:
        """Async variant of :meth:`record`.

        Default implementation runs the sync :meth:`record` in the event
        loop's default executor, so blocking I/O in ``record`` doesn't
        stall the loop. Override for native-async sinks.
        """
        await asyncio.get_running_loop().run_in_executor(None, self.record, event)

    def close(self) -> None:
        """Release any held resources. Default: no-op."""

    def __enter__(self) -> AuditLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


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
    """Append one JSON object per line to a file.

    Holds the file handle open for the lifetime of the log (rather than
    re-opening per record). Thread-safe. Always call :meth:`close` (or use
    as a context manager) to release the descriptor cleanly.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh: IO[str] | None = self.path.open("a", encoding="utf-8")

    def record(self, event: AuditEvent) -> None:
        line = event.to_json()
        with self._lock:
            if self._fh is None:
                raise RuntimeError("JsonLinesAuditLog is closed")
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                finally:
                    self._fh.close()
                    self._fh = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
