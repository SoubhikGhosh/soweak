"""Pipeline: executes a Policy against a Payload at a boundary."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from soweak.core.audit import AuditEvent, AuditLog
from soweak.core.detector import Signal
from soweak.core.enforcer import Action, Decision
from soweak.core.policy import Policy
from soweak.core.types import Boundary, Context, Payload


@dataclass
class Pipeline:
    """Run a Policy against payloads at one or more boundaries.

    A Pipeline is stateless apart from its policy and audit sink. Construct
    once at app startup, then call ``check_*`` helpers (or the lower-level
    ``run``) from anywhere in your code.
    """

    policy: Policy
    audit: AuditLog | None = None
    _last_signals: list[Signal] = field(default_factory=list, init=False, repr=False)

    def run(self, payload: Payload, ctx: Context | None = None) -> Decision:
        """Execute all rules attached to this payload's boundary.

        Rules are executed in declaration order. Each rule's enforcer receives
        only that rule's signals; if any rule blocks, the pipeline
        short-circuits. The returned Decision aggregates signals from every
        rule that ran.
        """
        ctx = ctx or Context()
        rules = self.policy.for_boundary(payload.boundary)
        all_signals: list[Signal] = []

        if not rules:
            decision = Decision.allow(payload)
            self._emit(ctx, payload.boundary, all_signals, decision)
            return decision

        last_decision: Decision | None = None
        for rule in rules:
            rule_signals: list[Signal] = []
            for det in rule.detectors:
                rule_signals.extend(det.inspect(payload, ctx))
            decision = rule.enforcer.decide(payload, rule_signals, ctx)
            all_signals.extend(rule_signals)
            payload = decision.payload  # allow redact/transform to carry forward
            last_decision = decision
            if decision.action == Action.BLOCK:
                break

        assert last_decision is not None
        final = replace(last_decision, signals=list(all_signals))
        self._emit(ctx, payload.boundary, all_signals, final)
        return final

    # ----- ergonomic helpers -------------------------------------------------

    def check_input(
        self, text: str, ctx: Context | None = None, **metadata: Any
    ) -> Decision:
        return self.run(Payload(Boundary.INPUT, text=text, metadata=metadata), ctx)

    def check_output(
        self, text: str, ctx: Context | None = None, **metadata: Any
    ) -> Decision:
        return self.run(Payload(Boundary.OUTPUT, text=text, metadata=metadata), ctx)

    def check_retrieval(
        self,
        documents: list[Any],
        ctx: Context | None = None,
        **metadata: Any,
    ) -> Decision:
        joined = "\n\n".join(_doc_text(d) for d in documents)
        return self.run(
            Payload(
                Boundary.RETRIEVAL,
                text=joined,
                raw=documents,
                metadata=metadata,
            ),
            ctx,
        )

    def check_tool_call(
        self,
        tool: str,
        arguments: dict[str, Any],
        ctx: Context | None = None,
        **metadata: Any,
    ) -> Decision:
        text = f"{tool}({arguments!r})"
        return self.run(
            Payload(
                Boundary.TOOL_CALL,
                text=text,
                raw={"tool": tool, "arguments": arguments},
                metadata=metadata,
            ),
            ctx,
        )

    # ----- internals ---------------------------------------------------------

    def _emit(
        self,
        ctx: Context,
        boundary: Boundary,
        signals: list[Signal],
        decision: Decision,
    ) -> None:
        if self.audit is None:
            return
        self.audit.record(
            AuditEvent(
                request_id=ctx.request_id,
                boundary=boundary,
                signals=list(signals),
                decision=decision,
            )
        )


def _doc_text(doc: Any) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for key in ("text", "page_content", "content", "body"):
            value = doc.get(key)
            if isinstance(value, str):
                return value
    text_attr = getattr(doc, "page_content", None) or getattr(doc, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return str(doc)
