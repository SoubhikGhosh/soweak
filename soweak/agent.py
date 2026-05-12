"""LLM06 — Excessive Agency: tool authorization framework.

The framework couples three things to every tool call:

1. **Scopes**: the tool requires certain capability scopes; the caller's
   :class:`Context` must grant them via ``ctx.metadata["granted_scopes"]``.
2. **Approval**: optionally requires a human approval handler to return
   ``True`` before the underlying function runs.
3. **Rate limit**: an in-process per-(tool, user) token bucket capping
   invocations per 60-second window.

Wrap tool functions with :func:`guarded_tool`. At the call site, set the
active context with :func:`authorize`. Tools called outside an
``authorize`` block raise :class:`PermissionError`.

Example::

    from soweak.agent import guarded_tool, authorize
    from soweak import Context

    @guarded_tool(scopes=["email:send"], approval="human", rate_limit_per_minute=5,
                  approval_handler=lambda call: input(f"Approve {call.tool}? [y/N] ") == "y")
    def send_email(to: str, subject: str, body: str) -> None:
        ...

    ctx = Context(user_id="alice",
                  metadata={"granted_scopes": frozenset({"email:send"})})
    with authorize(ctx):
        send_email("user@example.com", "hi", "hi")
"""

from __future__ import annotations

import contextvars
import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator, Protocol

from soweak.core.types import Context
from soweak.storage import InMemoryWindowStore, WindowStore


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class ApprovalRequired(PermissionError):
    """Raised when a guarded tool requires human approval and the handler
    rejected (or denied by default)."""


@dataclass(frozen=True)
class ToolCall:
    """The intent record an approval handler receives before the tool runs."""

    tool: str
    arguments: dict[str, Any]
    scopes: tuple[str, ...]
    user_id: str | None
    tenant_id: str | None


@dataclass(frozen=True)
class ToolCallEvent:
    """Audit record of a single guarded-tool invocation attempt."""

    timestamp: datetime
    tool: str
    arguments: dict[str, Any]
    scopes: tuple[str, ...]
    decision: str  # "allowed" | "denied_scope" | "denied_rate" | "denied_approval"
    user_id: str | None
    tenant_id: str | None
    reason: str = ""


class ApprovalHandler(Protocol):
    """Decide whether to approve a tool call awaiting human review."""

    def __call__(self, call: ToolCall) -> bool: ...


def _deny_all(call: ToolCall) -> bool:
    return False


# ---------------------------------------------------------------------------
# Authorization context
# ---------------------------------------------------------------------------


_current_ctx: contextvars.ContextVar[Context | None] = contextvars.ContextVar(
    "soweak_authorize_ctx", default=None
)

#: Well-known key on ``Context.metadata`` for an optional ``ToolCallEvent``
#: callback. If present, every guarded-tool invocation emits an event.
TOOL_AUDIT_KEY = "tool_audit_callback"

#: Well-known key on ``Context.metadata`` whose value is the
#: ``frozenset[str]`` of scopes granted to this context.
GRANTED_SCOPES_KEY = "granted_scopes"


@contextmanager
def authorize(ctx: Context) -> Iterator[Context]:
    """Set the active :class:`Context` for guarded-tool calls inside this block.

    Uses :mod:`contextvars`, so async tasks and threads spawned inside the
    block inherit the context naturally.
    """
    token = _current_ctx.set(ctx)
    try:
        yield ctx
    finally:
        _current_ctx.reset(token)


def current_context() -> Context | None:
    """Return the currently authorized context, or ``None`` if not in an
    :func:`authorize` block."""
    return _current_ctx.get()


# ---------------------------------------------------------------------------
# Rate limiting (in-process, per-tool per-user)
# ---------------------------------------------------------------------------


class _ToolRateLimiter:
    """Per-(tool, user) sliding-window limiter backed by a
    :class:`~soweak.storage.WindowStore`. Defaults to in-process; pass a
    persistent / multi-host store for shared limits."""

    def __init__(
        self,
        limit_per_minute: int,
        store: WindowStore | None = None,
        window_seconds: float = 60.0,
    ) -> None:
        if limit_per_minute <= 0:
            raise ValueError("rate limit must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.limit = limit_per_minute
        self._store: WindowStore = store or InMemoryWindowStore()
        self._window = window_seconds

    @property
    def store(self) -> WindowStore:
        return self._store

    def _key(self, tool: str, user: str) -> str:
        return f"tool:{tool}:user:{user}"

    def allow(self, tool: str, user: str) -> bool:
        now = time.time()
        key = self._key(tool, user)
        if self._store.count(key, now, self._window) >= self.limit:
            return False
        count = self._store.record(key, now, self._window)
        return count <= self.limit


# ---------------------------------------------------------------------------
# The decorator
# ---------------------------------------------------------------------------


def guarded_tool(
    scopes: Iterable[str] = (),
    approval: str = "auto",
    rate_limit_per_minute: int | None = None,
    approval_handler: ApprovalHandler | None = None,
    rate_limit_store: WindowStore | None = None,
    rate_limit_window_seconds: float = 60.0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that wraps a tool function with authorization, rate limiting
    and human-in-the-loop approval.

    Parameters:
      scopes: capability scopes the caller's context must grant.
      approval: ``"auto"`` (default — no human check) or ``"human"``.
      rate_limit_per_minute: optional max invocations per (tool, user) per
        rolling 60 second window.
      approval_handler: callable invoked when ``approval="human"``. Must
        return ``True`` to permit the call. Defaults to a deny-all handler.
      rate_limit_store: optional :class:`~soweak.storage.WindowStore` backing
        the rate limiter. Defaults to in-process; supply a SQLite or Redis
        store to share state across replicas.
      rate_limit_window_seconds: limiter window length. Default 60 seconds.

    Raises:
      PermissionError: when the active context lacks required scopes, is
        absent, or the rate limit is exceeded.
      ApprovalRequired: when ``approval="human"`` and the handler returns
        false.
    """
    required = frozenset(scopes)
    if approval not in ("auto", "human"):
        raise ValueError("approval must be 'auto' or 'human'")
    limiter = (
        _ToolRateLimiter(
            rate_limit_per_minute,
            store=rate_limit_store,
            window_seconds=rate_limit_window_seconds,
        )
        if rate_limit_per_minute
        else None
    )
    handler: ApprovalHandler = approval_handler or _deny_all

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = current_context()
            if ctx is None:
                _emit(None, fn, args, kwargs, "denied_scope", required, "no active context")
                raise PermissionError(
                    f"tool {fn.__name__} called outside an authorize() block"
                )

            granted = frozenset(ctx.metadata.get(GRANTED_SCOPES_KEY, frozenset()))
            missing = required - granted
            if missing:
                reason = f"missing scopes: {sorted(missing)}"
                _emit(ctx, fn, args, kwargs, "denied_scope", required, reason)
                raise PermissionError(
                    f"tool {fn.__name__} requires {sorted(required)}; missing {sorted(missing)}"
                )

            if limiter is not None:
                user = ctx.user_id or "anonymous"
                if not limiter.allow(fn.__name__, user):
                    reason = f"rate limit {limiter.limit}/min exceeded for user={user}"
                    _emit(ctx, fn, args, kwargs, "denied_rate", required, reason)
                    raise PermissionError(reason)

            if approval == "human":
                call = ToolCall(
                    tool=fn.__name__,
                    arguments={"args": args, "kwargs": kwargs},
                    scopes=tuple(sorted(required)),
                    user_id=ctx.user_id,
                    tenant_id=ctx.tenant_id,
                )
                if not handler(call):
                    _emit(ctx, fn, args, kwargs, "denied_approval", required, "approval rejected")
                    raise ApprovalRequired(
                        f"tool {fn.__name__} requires human approval"
                    )

            _emit(ctx, fn, args, kwargs, "allowed", required, "")
            return fn(*args, **kwargs)

        wrapper.__soweak_guarded__ = True  # type: ignore[attr-defined]
        wrapper.__soweak_scopes__ = required  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _emit(
    ctx: Context | None,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    decision: str,
    scopes: frozenset[str],
    reason: str,
) -> None:
    if ctx is None:
        return
    callback = ctx.metadata.get(TOOL_AUDIT_KEY)
    if not callable(callback):
        return
    event = ToolCallEvent(
        timestamp=datetime.now(timezone.utc),
        tool=fn.__name__,
        arguments={"args": args, "kwargs": kwargs},
        scopes=tuple(sorted(scopes)),
        decision=decision,
        user_id=ctx.user_id,
        tenant_id=ctx.tenant_id,
        reason=reason,
    )
    callback(event)


__all__ = [
    "ApprovalHandler",
    "ApprovalRequired",
    "GRANTED_SCOPES_KEY",
    "TOOL_AUDIT_KEY",
    "ToolCall",
    "ToolCallEvent",
    "authorize",
    "current_context",
    "guarded_tool",
]
