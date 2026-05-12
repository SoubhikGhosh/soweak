"""Storage backends + persisted budget/rate-limit integration tests."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

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
def counter_store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[object]:
    if request.param == "memory":
        store = InMemoryCounterStore()
        yield store
    else:
        store = SqliteCounterStore(tmp_path / "counter.db")
        try:
            yield store
        finally:
            store.close()


def test_counter_store_add_returns_new_total(counter_store) -> None:
    assert counter_store.add("k", 5) == 5
    assert counter_store.add("k", 3) == 8
    assert counter_store.get("k") == 8


def test_counter_store_add_respects_limit(counter_store) -> None:
    counter_store.add("k", 8)
    assert counter_store.add("k", 1, limit=10) == 9
    assert counter_store.add("k", 5, limit=10) is None
    assert counter_store.get("k") == 9


def test_counter_store_isolates_keys(counter_store) -> None:
    counter_store.add("a", 5)
    counter_store.add("b", 7)
    assert counter_store.get("a") == 5
    assert counter_store.get("b") == 7


def test_counter_store_reset_one_key(counter_store) -> None:
    counter_store.add("a", 5)
    counter_store.add("b", 7)
    counter_store.reset("a")
    assert counter_store.get("a") == 0
    assert counter_store.get("b") == 7


def test_counter_store_reset_all(counter_store) -> None:
    counter_store.add("a", 5)
    counter_store.add("b", 7)
    counter_store.reset()
    assert counter_store.get("a") == 0
    assert counter_store.get("b") == 0


# ---------------- SQLite counter persistence ----------------


def test_sqlite_counter_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "counter.db"
    with SqliteCounterStore(path) as store1:
        store1.add("alice", 100)
    with SqliteCounterStore(path) as store2:
        assert store2.get("alice") == 100


def test_sqlite_counter_close_is_idempotent(tmp_path: Path) -> None:
    store = SqliteCounterStore(tmp_path / "x.db")
    store.close()
    store.close()  # no raise
    with pytest.raises(RuntimeError, match="closed"):
        store.add("k", 1)


def test_sqlite_counter_context_manager(tmp_path: Path) -> None:
    with SqliteCounterStore(tmp_path / "ctx.db") as store:
        store.add("k", 7)
    # After exit, store is closed.
    with pytest.raises(RuntimeError, match="closed"):
        store.get("k")


# ---------------- WindowStore contract ----------------


@pytest.fixture(params=["memory", "sqlite"])
def window_store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[object]:
    if request.param == "memory":
        yield InMemoryWindowStore()
    else:
        store = SqliteWindowStore(tmp_path / "window.db")
        try:
            yield store
        finally:
            store.close()


def test_window_store_records_and_counts(window_store) -> None:
    t = 1_000.0
    assert window_store.record("u", t, 60.0) == 1
    assert window_store.record("u", t + 1, 60.0) == 2
    assert window_store.count("u", t + 2, 60.0) == 2


def test_window_store_drops_old_events(window_store) -> None:
    t = 1_000.0
    window_store.record("u", t, 60.0)
    assert window_store.record("u", t + 120, 60.0) == 1


def test_window_store_isolates_keys(window_store) -> None:
    t = 1_000.0
    window_store.record("a", t, 60.0)
    window_store.record("b", t, 60.0)
    assert window_store.count("a", t, 60.0) == 1


def test_window_store_reset(window_store) -> None:
    t = 1_000.0
    window_store.record("u", t, 60.0)
    window_store.reset("u")
    assert window_store.count("u", t, 60.0) == 0


def test_sqlite_window_close_blocks_use(tmp_path: Path) -> None:
    store = SqliteWindowStore(tmp_path / "w.db")
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.record("u", time.time(), 60.0)


# ---------------- Budget integration ----------------


def test_token_budget_uses_default_store() -> None:
    b = TokenBudget(limit=100)
    b.charge("alice", 30)
    assert b.consumed("alice") == 30


def test_token_budget_uses_supplied_store(tmp_path: Path) -> None:
    with SqliteCounterStore(tmp_path / "tb.db") as store:
        b1 = TokenBudget(limit=100, store=store)
        b1.charge("alice", 30)
        b2 = TokenBudget(limit=100, store=store)
        assert b2.consumed("alice") == 30


def test_token_budget_persists_across_restart(tmp_path: Path) -> None:
    db = tmp_path / "tb.db"
    with SqliteCounterStore(db) as s1:
        TokenBudget(limit=100, store=s1).charge("alice", 30)
    with SqliteCounterStore(db) as s2:
        b = TokenBudget(limit=100, store=s2)
        assert b.consumed("alice") == 30
        assert b.remaining("alice") == 70


def test_token_budget_atomic_limit_via_store() -> None:
    b = TokenBudget(limit=10)
    b.charge("u", 8)
    with pytest.raises(BudgetExceededError):
        b.charge("u", 5)
    assert b.consumed("u") == 8


def test_cost_budget_uses_supplied_store(tmp_path: Path) -> None:
    db = tmp_path / "cb.db"
    with SqliteCounterStore(db) as s1:
        b = CostBudget(limit_usd=1.0, store=s1)
        b.charge("alice", "gpt-4o-mini", input_tokens=1000, output_tokens=1000)
    with SqliteCounterStore(db) as s2:
        b2 = CostBudget(limit_usd=1.0, store=s2)
        assert b2.consumed("alice") > 0


# ---------------- RateLimiter with custom store ----------------


def test_rate_limiter_uses_default_window_store() -> None:
    r = RateLimiter(2)
    assert r.allow("u")
    assert r.allow("u")
    assert not r.allow("u")


def test_rate_limiter_with_sqlite_persists(tmp_path: Path) -> None:
    db = tmp_path / "rl.db"
    with SqliteWindowStore(db) as s1:
        r1 = RateLimiter(2, store=s1)
        assert r1.allow("u")
        assert r1.allow("u")
    with SqliteWindowStore(db) as s2:
        r2 = RateLimiter(2, store=s2)
        assert not r2.allow("u")


def test_rate_limiter_custom_window_seconds() -> None:
    r = RateLimiter(1, window_seconds=0.1)
    assert r.allow("u")
    assert not r.allow("u")
    time.sleep(0.15)
    assert r.allow("u")


# ---------------- Concurrency: thread safety claims ----------------


@pytest.mark.parametrize("store_factory", ["memory", "sqlite"])
def test_token_budget_concurrent_charges_never_exceed_limit(
    store_factory: str, tmp_path: Path
) -> None:
    """Hammer charge() from N threads; total charges <= limit at all times."""
    store_obj: object
    if store_factory == "memory":
        store_obj = InMemoryCounterStore()
    else:
        store_obj = SqliteCounterStore(tmp_path / "concurrent.db")
    try:
        budget = TokenBudget(limit=100, store=store_obj)  # type: ignore[arg-type]
        rejected = 0
        rejected_lock = threading.Lock()

        def attempt() -> None:
            nonlocal rejected
            try:
                budget.charge("u", 1)
            except BudgetExceededError:
                with rejected_lock:
                    rejected += 1

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(attempt) for _ in range(200)]
            for f in as_completed(futures):
                f.result()

        # Exactly 100 succeeded, exactly 100 rejected.
        assert budget.consumed("u") == 100
        assert rejected == 100
    finally:
        getattr(store_obj, "close", lambda: None)()


@pytest.mark.parametrize("store_factory", ["memory", "sqlite"])
def test_rate_limiter_concurrent_allow_count_is_correct(
    store_factory: str, tmp_path: Path
) -> None:
    """Hammer allow() from N threads; count of True returns equals the limit
    when total attempts >= limit (within one window)."""
    store_obj: object
    if store_factory == "memory":
        store_obj = InMemoryWindowStore()
    else:
        store_obj = SqliteWindowStore(tmp_path / "rlc.db")
    try:
        limit = 50
        limiter = RateLimiter(limit, store=store_obj, window_seconds=10.0)  # type: ignore[arg-type]
        wins = 0
        wins_lock = threading.Lock()

        def attempt() -> None:
            nonlocal wins
            if limiter.allow("u"):
                with wins_lock:
                    wins += 1

        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = [ex.submit(attempt) for _ in range(200)]
            for f in as_completed(futures):
                f.result()

        assert wins == limit
    finally:
        getattr(store_obj, "close", lambda: None)()
