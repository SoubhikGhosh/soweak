"""Storage backends + persisted budget/rate-limit integration tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from soweak import (
    BudgetExceededError,
    CostBudget,
    InMemoryCounterStore,
    InMemoryWindowStore,
    RateLimiter,
    SqliteCounterStore,
    SqliteWindowStore,
    TokenBudget,
)


# ---------------- CounterStore contract ----------------


@pytest.fixture(params=["memory", "sqlite"])
def counter_store(request, tmp_path: Path):
    if request.param == "memory":
        return InMemoryCounterStore()
    return SqliteCounterStore(tmp_path / "counter.db")


def test_counter_store_add_returns_new_total(counter_store):
    assert counter_store.add("k", 5) == 5
    assert counter_store.add("k", 3) == 8
    assert counter_store.get("k") == 8


def test_counter_store_add_respects_limit(counter_store):
    counter_store.add("k", 8)
    # Under limit: succeeds.
    assert counter_store.add("k", 1, limit=10) == 9
    # Over limit: returns None and does not mutate.
    assert counter_store.add("k", 5, limit=10) is None
    assert counter_store.get("k") == 9


def test_counter_store_isolates_keys(counter_store):
    counter_store.add("a", 5)
    counter_store.add("b", 7)
    assert counter_store.get("a") == 5
    assert counter_store.get("b") == 7


def test_counter_store_reset_one_key(counter_store):
    counter_store.add("a", 5)
    counter_store.add("b", 7)
    counter_store.reset("a")
    assert counter_store.get("a") == 0
    assert counter_store.get("b") == 7


def test_counter_store_reset_all(counter_store):
    counter_store.add("a", 5)
    counter_store.add("b", 7)
    counter_store.reset()
    assert counter_store.get("a") == 0
    assert counter_store.get("b") == 0


# ---------------- SQLite counter persistence ----------------


def test_sqlite_counter_survives_reopen(tmp_path: Path):
    path = tmp_path / "counter.db"
    store1 = SqliteCounterStore(path)
    store1.add("alice", 100)
    # Re-open as a new store on the same file.
    store2 = SqliteCounterStore(path)
    assert store2.get("alice") == 100


# ---------------- WindowStore contract ----------------


@pytest.fixture(params=["memory", "sqlite"])
def window_store(request, tmp_path: Path):
    if request.param == "memory":
        return InMemoryWindowStore()
    return SqliteWindowStore(tmp_path / "window.db")


def test_window_store_records_and_counts(window_store):
    t = 1_000.0
    assert window_store.record("u", t, 60.0) == 1
    assert window_store.record("u", t + 1, 60.0) == 2
    assert window_store.count("u", t + 2, 60.0) == 2


def test_window_store_drops_old_events(window_store):
    t = 1_000.0
    window_store.record("u", t, 60.0)
    # 120s later, the original event is outside the 60s window.
    assert window_store.record("u", t + 120, 60.0) == 1


def test_window_store_isolates_keys(window_store):
    t = 1_000.0
    window_store.record("a", t, 60.0)
    window_store.record("b", t, 60.0)
    assert window_store.count("a", t, 60.0) == 1


def test_window_store_reset(window_store):
    t = 1_000.0
    window_store.record("u", t, 60.0)
    window_store.reset("u")
    assert window_store.count("u", t, 60.0) == 0


# ---------------- Budget integration ----------------


def test_token_budget_uses_default_store():
    b = TokenBudget(limit=100)
    b.charge("alice", 30)
    assert b.consumed("alice") == 30


def test_token_budget_uses_supplied_store(tmp_path: Path):
    store = SqliteCounterStore(tmp_path / "tb.db")
    b1 = TokenBudget(limit=100, store=store)
    b1.charge("alice", 30)
    # Recreate budget on the same store — state survives.
    b2 = TokenBudget(limit=100, store=store)
    assert b2.consumed("alice") == 30


def test_token_budget_persists_across_restart(tmp_path: Path):
    db = tmp_path / "tb.db"
    b1 = TokenBudget(limit=100, store=SqliteCounterStore(db))
    b1.charge("alice", 30)
    # Imagine the process restarts: brand-new store + budget on the same file.
    b2 = TokenBudget(limit=100, store=SqliteCounterStore(db))
    assert b2.consumed("alice") == 30
    assert b2.remaining("alice") == 70


def test_token_budget_atomic_limit_via_store():
    """The store's add(limit=...) protocol prevents over-charge."""
    b = TokenBudget(limit=10)
    b.charge("u", 8)
    with pytest.raises(BudgetExceededError):
        b.charge("u", 5)
    # State unchanged on rejection.
    assert b.consumed("u") == 8


def test_cost_budget_uses_supplied_store(tmp_path: Path):
    store = SqliteCounterStore(tmp_path / "cb.db")
    b = CostBudget(limit_usd=1.0, store=store)
    b.charge("alice", "gpt-4o-mini", input_tokens=1000, output_tokens=1000)
    b2 = CostBudget(limit_usd=1.0, store=SqliteCounterStore(tmp_path / "cb.db"))
    assert b2.consumed("alice") > 0


# ---------------- RateLimiter with custom store ----------------


def test_rate_limiter_uses_default_window_store():
    r = RateLimiter(2)
    assert r.allow("u")
    assert r.allow("u")
    assert not r.allow("u")


def test_rate_limiter_with_sqlite_persists(tmp_path: Path):
    store = SqliteWindowStore(tmp_path / "rl.db")
    r1 = RateLimiter(2, store=store)
    assert r1.allow("u")
    assert r1.allow("u")
    # New limiter on the same store sees the existing events.
    r2 = RateLimiter(2, store=SqliteWindowStore(tmp_path / "rl.db"))
    assert not r2.allow("u")


def test_rate_limiter_custom_window_seconds(tmp_path: Path):
    """A shorter window means rejected requests recover sooner."""
    r = RateLimiter(1, window_seconds=0.1)
    assert r.allow("u")
    assert not r.allow("u")
    time.sleep(0.15)
    assert r.allow("u")
