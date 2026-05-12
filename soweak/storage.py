"""Pluggable storage backends for budgets and rate limits.

Two abstract interfaces:

* :class:`CounterStore` — atomic counter operations for token / cost
  budgets. ``add(key, delta, limit=None)`` returns the new total, or
  ``None`` when the operation would exceed ``limit`` (no-op in that case).
* :class:`WindowStore` — sliding-window event store for rate limits.
  ``record(key, timestamp, window_seconds)`` adds an event and returns
  the count of events in the trailing window.

Built-in implementations:

* :class:`InMemoryCounterStore` / :class:`InMemoryWindowStore` — default
  in-process, thread-safe. State resets on restart.
* :class:`SqliteCounterStore` / :class:`SqliteWindowStore` — file-backed,
  survive restarts; safe for a single host. Connections are held open for
  the lifetime of the store, lock-serialised, and explicitly closed via
  :meth:`close` (or by using the store as a context manager). For
  multi-host deployments swap in a Redis backend by subclassing either
  ABC.

Pass a store to :class:`TokenBudget`, :class:`CostBudget`,
:class:`RateLimiter`, and :class:`RateLimitEnforcer` to share state
across replicas or persist across restarts.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any


# ---------------------------------------------------------------------------
# Counter store (for budgets)
# ---------------------------------------------------------------------------


class CounterStore(ABC):
    """Atomic counters keyed by string."""

    @abstractmethod
    def add(self, key: str, delta: float, limit: float | None = None) -> float | None:
        """Add ``delta`` to the counter at ``key``.

        If ``limit`` is provided and the resulting total would exceed
        ``limit``, do nothing and return ``None``. Otherwise return the new
        total.
        """

    @abstractmethod
    def get(self, key: str) -> float:
        """Return the current value at ``key`` (0.0 if unset)."""

    @abstractmethod
    def reset(self, key: str | None = None) -> None:
        """Reset ``key`` to 0, or every key when ``key`` is ``None``."""

    def close(self) -> None:
        """Release any held resources. Default: no-op."""

    def __enter__(self) -> CounterStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class InMemoryCounterStore(CounterStore):
    """Thread-safe in-process dict. Default for budgets."""

    def __init__(self) -> None:
        self._values: dict[str, float] = {}
        self._lock = threading.Lock()

    def add(self, key: str, delta: float, limit: float | None = None) -> float | None:
        with self._lock:
            new = self._values.get(key, 0.0) + delta
            if limit is not None and new > limit:
                return None
            self._values[key] = new
            return new

    def get(self, key: str) -> float:
        with self._lock:
            return self._values.get(key, 0.0)

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._values.clear()
            else:
                self._values.pop(key, None)


class SqliteCounterStore(CounterStore):
    """SQLite-backed counters with a single, lock-serialised connection.

    Survives process restarts; safe for a single host. Multi-host deployments
    need a different backend (Redis, etc.). Always call :meth:`close` (or
    use as a context manager) to release the file handle cleanly.
    """

    _SCHEMA = (
        "CREATE TABLE IF NOT EXISTS soweak_counters ("
        "key TEXT PRIMARY KEY, value REAL NOT NULL DEFAULT 0"
        ")"
    )

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._closed = False
        # One persistent connection; concurrent operations serialised via lock.
        self._conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we use explicit BEGIN/COMMIT.
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(self._SCHEMA)

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("SqliteCounterStore is closed")

    def add(self, key: str, delta: float, limit: float | None = None) -> float | None:
        with self._lock:
            self._check()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT value FROM soweak_counters WHERE key = ?", (key,)
                ).fetchone()
                current = float(row[0]) if row else 0.0
                new = current + delta
                if limit is not None and new > limit:
                    self._conn.execute("ROLLBACK")
                    return None
                self._conn.execute(
                    "INSERT INTO soweak_counters(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, new),
                )
                self._conn.execute("COMMIT")
                return new
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def get(self, key: str) -> float:
        with self._lock:
            self._check()
            row = self._conn.execute(
                "SELECT value FROM soweak_counters WHERE key = ?", (key,)
            ).fetchone()
            return float(row[0]) if row else 0.0

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            self._check()
            if key is None:
                self._conn.execute("DELETE FROM soweak_counters")
            else:
                self._conn.execute(
                    "DELETE FROM soweak_counters WHERE key = ?", (key,)
                )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __del__(self) -> None:
        # Best-effort: don't rely on this; explicit close() is preferred.
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Window store (for rate limits)
# ---------------------------------------------------------------------------


class WindowStore(ABC):
    """Sliding-window event store keyed by string."""

    @abstractmethod
    def record(self, key: str, timestamp: float, window_seconds: float) -> int:
        """Add an event at ``timestamp`` and return the count of events
        whose timestamp lies in ``(timestamp - window_seconds, timestamp]``."""

    @abstractmethod
    def count(self, key: str, now: float, window_seconds: float) -> int:
        """Return the count of events in the current trailing window
        without adding a new event."""

    @abstractmethod
    def reset(self, key: str | None = None) -> None: ...

    def close(self) -> None:
        """Release any held resources. Default: no-op."""

    def __enter__(self) -> WindowStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class InMemoryWindowStore(WindowStore):
    """In-process timestamp lists. Default for rate limits."""

    def __init__(self) -> None:
        self._timestamps: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def record(self, key: str, timestamp: float, window_seconds: float) -> int:
        with self._lock:
            bucket = self._timestamps.setdefault(key, [])
            cutoff = timestamp - window_seconds
            bucket[:] = [t for t in bucket if t > cutoff]
            bucket.append(timestamp)
            return len(bucket)

    def count(self, key: str, now: float, window_seconds: float) -> int:
        with self._lock:
            bucket = self._timestamps.get(key, [])
            cutoff = now - window_seconds
            return sum(1 for t in bucket if t > cutoff)

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._timestamps.clear()
            else:
                self._timestamps.pop(key, None)


class SqliteWindowStore(WindowStore):
    """SQLite-backed sliding-window store with a single, lock-serialised
    connection."""

    _SCHEMA = (
        "CREATE TABLE IF NOT EXISTS soweak_events ("
        "key TEXT NOT NULL, ts REAL NOT NULL"
        ");"
        "CREATE INDEX IF NOT EXISTS idx_soweak_events_key_ts ON soweak_events(key, ts)"
    )

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._closed = False
        self._conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(self._SCHEMA)

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("SqliteWindowStore is closed")

    def record(self, key: str, timestamp: float, window_seconds: float) -> int:
        cutoff = timestamp - window_seconds
        with self._lock:
            self._check()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "DELETE FROM soweak_events WHERE key = ? AND ts <= ?",
                    (key, cutoff),
                )
                self._conn.execute(
                    "INSERT INTO soweak_events(key, ts) VALUES(?, ?)", (key, timestamp)
                )
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM soweak_events WHERE key = ? AND ts > ?",
                    (key, cutoff),
                ).fetchone()
                self._conn.execute("COMMIT")
                return int(row[0]) if row else 0
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def count(self, key: str, now: float, window_seconds: float) -> int:
        cutoff = now - window_seconds
        with self._lock:
            self._check()
            row = self._conn.execute(
                "SELECT COUNT(*) FROM soweak_events WHERE key = ? AND ts > ?",
                (key, cutoff),
            ).fetchone()
            return int(row[0]) if row else 0

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            self._check()
            if key is None:
                self._conn.execute("DELETE FROM soweak_events")
            else:
                self._conn.execute("DELETE FROM soweak_events WHERE key = ?", (key,))

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "CounterStore",
    "InMemoryCounterStore",
    "InMemoryWindowStore",
    "SqliteCounterStore",
    "SqliteWindowStore",
    "WindowStore",
]
