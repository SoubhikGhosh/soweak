"""LLM10 budget and rate-limit tests."""

from __future__ import annotations

import pytest

from soweak import (
    Action,
    BudgetEnforcer,
    BudgetExceededError,
    Boundary,
    Context,
    CostBudget,
    ModelPricing,
    Payload,
    RateLimitEnforcer,
    RateLimiter,
    TokenBudget,
)


# ---------------- TokenBudget ----------------


def test_token_budget_tracks_per_scope():
    b = TokenBudget(limit=100)
    b.charge("alice", 30)
    b.charge("alice", 50)
    b.charge("bob", 90)
    assert b.consumed("alice") == 80
    assert b.consumed("bob") == 90
    assert b.remaining("alice") == 20


def test_token_budget_raises_when_exceeded():
    b = TokenBudget(limit=10)
    b.charge("u", 8)
    with pytest.raises(BudgetExceededError) as exc:
        b.charge("u", 5)
    assert exc.value.scope == "u"
    assert exc.value.limit == 10
    assert exc.value.attempted == 13
    # State unchanged on failure
    assert b.consumed("u") == 8


def test_token_budget_rejects_negative():
    b = TokenBudget(limit=10)
    with pytest.raises(ValueError):
        b.charge("u", -1)


def test_token_budget_reset():
    b = TokenBudget(limit=10)
    b.charge("a", 5)
    b.charge("b", 5)
    b.reset("a")
    assert b.consumed("a") == 0
    assert b.consumed("b") == 5
    b.reset()
    assert b.consumed("b") == 0


def test_token_budget_invalid_limit():
    with pytest.raises(ValueError):
        TokenBudget(limit=0)


# ---------------- CostBudget ----------------


def test_cost_budget_with_default_pricing():
    b = CostBudget(limit_usd=1.0)
    cost = b.charge("alice", "gpt-4o-mini", input_tokens=1000, output_tokens=1000)
    assert cost == pytest.approx(0.00015 + 0.00060)


def test_cost_budget_raises_when_exceeded():
    b = CostBudget(limit_usd=0.001)
    with pytest.raises(BudgetExceededError):
        b.charge("u", "gpt-4o", input_tokens=10_000, output_tokens=10_000)


def test_cost_budget_unknown_model():
    b = CostBudget(limit_usd=10.0)
    with pytest.raises(KeyError, match="no pricing"):
        b.charge("u", "non-existent-model", 100, 100)


def test_cost_budget_custom_pricing_registration():
    b = CostBudget(limit_usd=1.0)
    b.register_pricing("custom-llm", ModelPricing(input_per_1k=0.001, output_per_1k=0.002))
    cost = b.charge("u", "custom-llm", input_tokens=1000, output_tokens=1000)
    assert cost == pytest.approx(0.003)


# ---------------- BudgetEnforcer ----------------


def test_budget_enforcer_blocks_when_exhausted():
    b = TokenBudget(limit=10)
    b.charge("alice", 10)
    enf = BudgetEnforcer(b, scope_attr="user_id")
    ctx = Context(user_id="alice")
    d = enf.decide(Payload(Boundary.INPUT, text="x"), [], ctx)
    assert d.action is Action.BLOCK
    assert "exhausted" in d.reason


def test_budget_enforcer_allows_when_room_left():
    b = TokenBudget(limit=100)
    b.charge("alice", 50)
    enf = BudgetEnforcer(b, scope_attr="user_id")
    ctx = Context(user_id="alice")
    d = enf.decide(Payload(Boundary.INPUT, text="x"), [], ctx)
    assert d.action is Action.ALLOW


def test_budget_enforcer_uses_default_scope_when_attr_missing():
    b = TokenBudget(limit=10)
    b.charge("default", 10)
    enf = BudgetEnforcer(b, scope_attr="user_id")
    ctx = Context()  # no user_id
    d = enf.decide(Payload(Boundary.INPUT, text="x"), [], ctx)
    assert d.action is Action.BLOCK


# ---------------- RateLimitEnforcer ----------------


def test_rate_limit_enforcer_blocks_after_limit():
    enf = RateLimitEnforcer(requests_per_minute=2, scope_attr="user_id")
    ctx = Context(user_id="alice")
    p = Payload(Boundary.INPUT, text="x")
    assert enf.decide(p, [], ctx).action is Action.ALLOW
    assert enf.decide(p, [], ctx).action is Action.ALLOW
    assert enf.decide(p, [], ctx).action is Action.BLOCK


def test_rate_limit_isolates_per_scope():
    enf = RateLimitEnforcer(requests_per_minute=1)
    p = Payload(Boundary.INPUT, text="x")
    assert enf.decide(p, [], Context(user_id="a")).action is Action.ALLOW
    assert enf.decide(p, [], Context(user_id="b")).action is Action.ALLOW
    assert enf.decide(p, [], Context(user_id="a")).action is Action.BLOCK


def test_rate_limiter_standalone():
    r = RateLimiter(requests_per_minute=2)
    assert r.allow("u")
    assert r.allow("u")
    assert not r.allow("u")
