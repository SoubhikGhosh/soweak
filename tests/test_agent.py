"""LLM06 tool authorization tests."""

from __future__ import annotations

import pytest

from soweak import (
    ApprovalRequired,
    Context,
    ToolCall,
    ToolCallEvent,
    authorize,
    current_context,
    guarded_tool,
)
from soweak.agent import GRANTED_SCOPES_KEY, TOOL_AUDIT_KEY


def _ctx(scopes=(), **kw):
    return Context(metadata={GRANTED_SCOPES_KEY: frozenset(scopes), **kw.pop("metadata", {})}, **kw)


def test_call_outside_authorize_raises():
    @guarded_tool(scopes=["x"])
    def t():
        return "ok"

    with pytest.raises(PermissionError, match="outside an authorize"):
        t()


def test_authorize_sets_current_context():
    ctx = _ctx()
    assert current_context() is None
    with authorize(ctx):
        assert current_context() is ctx
    assert current_context() is None


def test_scope_grant_allows_call():
    @guarded_tool(scopes=["email:send"])
    def send():
        return "sent"

    with authorize(_ctx(scopes=["email:send"])):
        assert send() == "sent"


def test_missing_scope_raises():
    @guarded_tool(scopes=["email:send", "db:write"])
    def both():
        return "ok"

    with authorize(_ctx(scopes=["email:send"])):
        with pytest.raises(PermissionError, match="db:write"):
            both()


def test_no_scopes_required_works():
    @guarded_tool()
    def noop():
        return 42

    with authorize(_ctx()):
        assert noop() == 42


def test_human_approval_required_and_handler_runs():
    seen: list[ToolCall] = []

    def handler(call: ToolCall) -> bool:
        seen.append(call)
        return True

    @guarded_tool(scopes=["x"], approval="human", approval_handler=handler)
    def act(x: int) -> int:
        return x * 2

    with authorize(_ctx(scopes=["x"], user_id="alice")):
        assert act(7) == 14

    assert len(seen) == 1
    assert seen[0].tool == "act"
    assert seen[0].user_id == "alice"


def test_human_approval_default_denies():
    @guarded_tool(scopes=["x"], approval="human")  # default handler denies
    def act():
        return "ok"

    with authorize(_ctx(scopes=["x"])):
        with pytest.raises(ApprovalRequired):
            act()


def test_rate_limit_enforced():
    @guarded_tool(rate_limit_per_minute=2)
    def t():
        return "ok"

    with authorize(_ctx(user_id="alice")):
        assert t() == "ok"
        assert t() == "ok"
        with pytest.raises(PermissionError, match="rate limit"):
            t()


def test_audit_callback_receives_events():
    events: list[ToolCallEvent] = []

    @guarded_tool(scopes=["x"])
    def t():
        return "ok"

    ctx = Context(metadata={
        GRANTED_SCOPES_KEY: frozenset({"x"}),
        TOOL_AUDIT_KEY: events.append,
    })
    with authorize(ctx):
        t()

    assert len(events) == 1
    assert events[0].decision == "allowed"
    assert events[0].tool == "t"


def test_audit_callback_records_denial():
    events: list[ToolCallEvent] = []
    ctx = Context(metadata={
        GRANTED_SCOPES_KEY: frozenset(),
        TOOL_AUDIT_KEY: events.append,
    })

    @guarded_tool(scopes=["x"])
    def t():
        return "ok"

    with authorize(ctx):
        with pytest.raises(PermissionError):
            t()

    assert events[-1].decision == "denied_scope"


def test_invalid_approval_value_rejected():
    with pytest.raises(ValueError, match="auto"):
        guarded_tool(approval="maybe")


def test_decorator_preserves_signature():
    @guarded_tool()
    def doubler(x: int) -> int:
        """double x"""
        return x * 2

    assert doubler.__name__ == "doubler"
    assert "double x" in (doubler.__doc__ or "")
    assert getattr(doubler, "__soweak_guarded__", False)


def test_authorize_is_async_safe():
    """contextvars should isolate per task."""
    import asyncio

    @guarded_tool(scopes=["x"])
    def t():
        return current_context().user_id

    async def run(name):
        with authorize(_ctx(scopes=["x"], user_id=name)):
            await asyncio.sleep(0)
            return t()

    async def main():
        results = await asyncio.gather(run("alice"), run("bob"))
        return results

    assert asyncio.run(main()) == ["alice", "bob"]
