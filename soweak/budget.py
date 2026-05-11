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

from soweak.core.detector import Signal
from soweak.core.enforcer import Action, Decision, Enforcer
from soweak.core.types import Context, OwaspCategory, Payload, Severity


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
    request). Thread-safe."""

    def __init__(self, limit: int, name: str = "token-budget") -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self._limit = limit
        self._name = name
        self._consumed: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def limit(self) -> int:
        return self._limit

    def charge(self, scope: str, tokens: int) -> int:
        """Charge ``tokens`` against ``scope``. Returns new total. Raises
        :class:`BudgetExceededError` if the charge would exceed the limit."""
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        with self._lock:
            current = self._consumed.get(scope, 0)
            new = current + tokens
            if new > self._limit:
                raise BudgetExceededError(self._name, scope, self._limit, new)
            self._consumed[scope] = new
            return new

    def consumed(self, scope: str) -> int:
        with self._lock:
            return self._consumed.get(scope, 0)

    def remaining(self, scope: str) -> int:
        with self._lock:
            return max(0, self._limit - self._consumed.get(scope, 0))

    def reset(self, scope: str | None = None) -> None:
        with self._lock:
            if scope is None:
                self._consumed.clear()
            else:
                self._consumed.pop(scope, None)


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
    ) -> None:
        if limit_usd <= 0:
            raise ValueError("limit_usd must be positive")
        self._limit = float(limit_usd)
        self._name = name
        self._pricing: dict[str, ModelPricing] = dict(pricing) if pricing is not None else dict(DEFAULT_PRICING)
        self._consumed: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def limit(self) -> float:
        return self._limit

    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        with self._lock:
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
        with self._lock:
            new = self._consumed.get(scope, 0.0) + cost
            if new > self._limit:
                raise BudgetExceededError(self._name, scope, self._limit, new)
            self._consumed[scope] = new
            return new

    def consumed(self, scope: str) -> float:
        with self._lock:
            return self._consumed.get(scope, 0.0)

    def remaining(self, scope: str) -> float:
        with self._lock:
            return max(0.0, self._limit - self._consumed.get(scope, 0.0))

    def reset(self, scope: str | None = None) -> None:
        with self._lock:
            if scope is None:
                self._consumed.clear()
            else:
                self._consumed.pop(scope, None)


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
        budget: TokenBudget | CostBudget,
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
    """Sliding-window in-process rate limiter. Per (scope) per 60s."""

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._limit = requests_per_minute
        self._timestamps: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    def allow(self, scope: str) -> bool:
        now = time.time()
        with self._lock:
            bucket = self._timestamps.setdefault(scope, [])
            cutoff = now - 60.0
            bucket[:] = [t for t in bucket if t > cutoff]
            if len(bucket) >= self._limit:
                return False
            bucket.append(now)
            return True


class RateLimitEnforcer(Enforcer):
    """Block when the caller's scope has used too many requests this window."""

    def __init__(
        self,
        requests_per_minute: int,
        scope_attr: str = "user_id",
        name: str = "rate-limit",
    ) -> None:
        self._limiter = RateLimiter(requests_per_minute)
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
    "BudgetEnforcer",
    "BudgetExceededError",
    "CostBudget",
    "DEFAULT_PRICING",
    "ModelPricing",
    "RateLimitEnforcer",
    "RateLimiter",
    "TokenBudget",
]
