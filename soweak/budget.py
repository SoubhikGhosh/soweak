"""LLM10 — Unbounded Consumption: token and cost budgets, rate limits.

These are stateful trackers and enforcers. Instantiate once per scope
(per-process, per-user, per-tenant) and call ``charge`` from the call site
that actually consumes tokens (after the LLM call returns).

Example::

    from soweak import Pipeline, PolicyBuilder
    from soweak.budget import TokenBudget, BudgetEnforcer

    budget = TokenBudget(limit=1_000_000)  # 1M tokens per scope

    pipeline = Pipeline(
        PolicyBuilder()
        .on_input("budget-gate")
            .enforce(BudgetEnforcer(budget, scope_attr="user_id"))
        .build()
    )

    # Pre-call: pipeline blocks if budget is already exhausted.
    pipeline.check_input(prompt, ctx)

    # Post-call: charge actual usage.
    budget.charge(scope=ctx.user_id, tokens=response.usage.total_tokens)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from typing import Protocol, runtime_checkable

from soweak.core.detector import Signal
from soweak.core.enforcer import Action, Decision, Enforcer
from soweak.core.types import Context, OwaspCategory, Payload, Severity
from soweak.storage import (
    CounterStore,
    InMemoryCounterStore,
    InMemoryWindowStore,
    WindowStore,
)


# ---------------------------------------------------------------------------
# Budget protocol — every budget exposes the same minimal surface.
# ---------------------------------------------------------------------------


@runtime_checkable
class Budget(Protocol):
    """Protocol shared by :class:`TokenBudget`, :class:`CostBudget`, and any
    user-defined budget (e.g., request count, byte count).

    Enforcers that want to gate on "is this scope out of budget?" should
    type their parameter as :class:`Budget` rather than a concrete class so
    new budget types compose cleanly.
    """

    @property
    def name(self) -> str: ...

    def consumed(self, scope: str) -> float: ...

    def remaining(self, scope: str) -> float: ...

    def reset(self, scope: str | None = None) -> None: ...


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BudgetExceededError(RuntimeError):
    """Raised by ``Budget.charge`` when a charge would exceed the limit."""

    def __init__(self, budget_name: str, scope: str, limit: float, attempted: float) -> None:
        super().__init__(
            f"budget {budget_name!r} exceeded for scope {scope!r}: "
            f"attempted {attempted}, limit {limit}"
        )
        self.budget_name = budget_name
        self.scope = scope
        self.limit = limit
        self.attempted = attempted


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


class TokenBudget:
    """Tracks integer token consumption per scope key (e.g. user, session,
    request).

    Backed by a pluggable :class:`~soweak.storage.CounterStore`. Defaults to
    in-process :class:`InMemoryCounterStore`; swap in
    :class:`~soweak.storage.SqliteCounterStore` (or your own Redis/Postgres
    backend) to persist across restarts and share across replicas.
    """

    def __init__(
        self,
        limit: int,
        name: str = "token-budget",
        store: CounterStore | None = None,
    ) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self._limit = limit
        self._name = name
        self._store: CounterStore = store or InMemoryCounterStore()

    @property
    def name(self) -> str:
        return self._name

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def store(self) -> CounterStore:
        return self._store

    def _key(self, scope: str) -> str:
        return f"{self._name}:{scope}"

    def charge(self, scope: str, tokens: int) -> int:
        """Charge ``tokens`` against ``scope``. Returns new total. Raises
        :class:`BudgetExceededError` if the charge would exceed the limit."""
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        new = self._store.add(self._key(scope), float(tokens), limit=float(self._limit))
        if new is None:
            attempted = self._store.get(self._key(scope)) + tokens
            raise BudgetExceededError(self._name, scope, self._limit, attempted)
        return int(new)

    def consumed(self, scope: str) -> int:
        return int(self._store.get(self._key(scope)))

    def remaining(self, scope: str) -> int:
        return max(0, self._limit - self.consumed(scope))

    def reset(self, scope: str | None = None) -> None:
        if scope is None:
            self._store.reset()
        else:
            self._store.reset(self._key(scope))


# ---------------------------------------------------------------------------
# Cost budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPricing:
    """Per-1k-token pricing in USD."""

    input_per_1k: float
    output_per_1k: float


#: Approximate prices for common models. Update per provider terms.
DEFAULT_PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(input_per_1k=0.0050, output_per_1k=0.0150),
    "gpt-4o-mini": ModelPricing(input_per_1k=0.00015, output_per_1k=0.00060),
    "claude-sonnet-4-5": ModelPricing(input_per_1k=0.0030, output_per_1k=0.0150),
    "claude-haiku-4-5": ModelPricing(input_per_1k=0.0008, output_per_1k=0.0040),
    "gemini-1.5-pro": ModelPricing(input_per_1k=0.0035, output_per_1k=0.0105),
    "gemini-1.5-flash": ModelPricing(input_per_1k=0.000075, output_per_1k=0.00030),
}


class CostBudget:
    """Tracks USD spend per scope, given a per-model pricing table.

    Pricing missing for a requested model raises :class:`KeyError`.
    """

    def __init__(
        self,
        limit_usd: float,
        pricing: Mapping[str, ModelPricing] | None = None,
        name: str = "cost-budget",
        store: CounterStore | None = None,
    ) -> None:
        if limit_usd <= 0:
            raise ValueError("limit_usd must be positive")
        self._limit = float(limit_usd)
        self._name = name
        self._pricing: dict[str, ModelPricing] = dict(pricing) if pricing is not None else dict(DEFAULT_PRICING)
        self._pricing_lock = threading.Lock()
        self._store: CounterStore = store or InMemoryCounterStore()

    @property
    def name(self) -> str:
        return self._name

    @property
    def limit(self) -> float:
        return self._limit

    @property
    def store(self) -> CounterStore:
        return self._store

    def _key(self, scope: str) -> str:
        return f"{self._name}:{scope}"

    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        with self._pricing_lock:
            self._pricing[model] = pricing

    def _cost_of(self, model: str, input_tokens: int, output_tokens: int) -> float:
        try:
            rate = self._pricing[model]
        except KeyError as e:
            raise KeyError(
                f"no pricing registered for model {model!r}; call register_pricing()"
            ) from e
        return (input_tokens / 1000.0) * rate.input_per_1k + (
            output_tokens / 1000.0
        ) * rate.output_per_1k

    def charge(
        self, scope: str, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        cost = self._cost_of(model, input_tokens, output_tokens)
        new = self._store.add(self._key(scope), cost, limit=self._limit)
        if new is None:
            attempted = self._store.get(self._key(scope)) + cost
            raise BudgetExceededError(self._name, scope, self._limit, attempted)
        return float(new)

    def consumed(self, scope: str) -> float:
        return float(self._store.get(self._key(scope)))

    def remaining(self, scope: str) -> float:
        return max(0.0, self._limit - self.consumed(scope))

    def reset(self, scope: str | None = None) -> None:
        if scope is None:
            self._store.reset()
        else:
            self._store.reset(self._key(scope))


# ---------------------------------------------------------------------------
# Enforcers
# ---------------------------------------------------------------------------


class BudgetEnforcer(Enforcer):
    """Block when a budget is already exhausted for the request's scope.

    Reads the scope key from ``getattr(ctx, scope_attr)``; falls back to the
    string ``"default"``. Does not charge — pair with explicit ``budget.charge``
    after your LLM call returns.
    """

    def __init__(
        self,
        budget: Budget,
        scope_attr: str = "user_id",
        name: str | None = None,
    ) -> None:
        self._budget = budget
        self._scope_attr = scope_attr
        self._name = name or f"budget[{budget.name}]"

    @property
    def name(self) -> str:
        return self._name

    def decide(
        self, payload: Payload, signals: list[Signal], ctx: Context
    ) -> Decision:
        scope = getattr(ctx, self._scope_attr, None) or "default"
        remaining = self._budget.remaining(scope)
        if remaining <= 0:
            return Decision(
                Action.BLOCK,
                payload,
                list(signals),
                reason=f"budget {self._budget.name!r} exhausted for {scope!r}",
                metadata={
                    "budget": self._budget.name,
                    "scope": scope,
                    "remaining": remaining,
                },
            )
        if signals:
            return Decision(Action.WARN, payload, list(signals))
        return Decision.allow(payload)


class RateLimiter:
    """Sliding-window rate limiter, backed by a pluggable :class:`WindowStore`.

    Defaults to :class:`InMemoryWindowStore`; swap in
    :class:`~soweak.storage.SqliteWindowStore` for restart-survival or a
    custom Redis-backed store for multi-host.
    """

    def __init__(
        self,
        requests_per_minute: int,
        store: WindowStore | None = None,
        window_seconds: float = 60.0,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._limit = requests_per_minute
        self._window = window_seconds
        self._store: WindowStore = store or InMemoryWindowStore()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def store(self) -> WindowStore:
        return self._store

    def allow(self, scope: str) -> bool:
        now = time.time()
        # First check the count under the current window; only record when
        # there's room. This avoids inflating the bucket on rejected requests.
        if self._store.count(scope, now, self._window) >= self._limit:
            return False
        count = self._store.record(scope, now, self._window)
        return count <= self._limit


class RateLimitEnforcer(Enforcer):
    """Block when the caller's scope has used too many requests this window."""

    def __init__(
        self,
        requests_per_minute: int,
        scope_attr: str = "user_id",
        name: str = "rate-limit",
        store: WindowStore | None = None,
        window_seconds: float = 60.0,
    ) -> None:
        self._limiter = RateLimiter(
            requests_per_minute, store=store, window_seconds=window_seconds
        )
        self._scope_attr = scope_attr
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def limit(self) -> int:
        return self._limiter.limit

    def decide(
        self, payload: Payload, signals: list[Signal], ctx: Context
    ) -> Decision:
        scope = getattr(ctx, self._scope_attr, None) or "default"
        if not self._limiter.allow(scope):
            return Decision(
                Action.BLOCK,
                payload,
                list(signals),
                reason=f"rate limit {self._limiter.limit}/min exceeded for {scope!r}",
            )
        if signals:
            return Decision(Action.WARN, payload, list(signals))
        return Decision.allow(payload)


__all__ = [
    "Budget",
    "BudgetEnforcer",
    "BudgetExceededError",
    "CostBudget",
    "DEFAULT_PRICING",
    "ModelPricing",
    "RateLimitEnforcer",
    "RateLimiter",
    "TokenBudget",
]
