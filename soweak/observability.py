"""OpenTelemetry exporter for soweak audit events.

Install with ``pip install soweak[otel]``.

Every :class:`~soweak.AuditEvent` becomes a child span with the boundary
in its name (``soweak.input``, ``soweak.output``, etc.). Signals are
attached as span events with severity/category/confidence attributes.
The final decision becomes attributes on the span.

Use it like any other :class:`~soweak.AuditLog`::

    from opentelemetry import trace
    from soweak import Pipeline
    from soweak.observability import OpenTelemetryAuditLog

    pipeline = Pipeline(policy, audit=OpenTelemetryAuditLog())
"""

from __future__ import annotations

from typing import Any

from soweak.core.audit import AuditEvent, AuditLog


def _require_otel() -> Any:
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError(
            "OpenTelemetry support requires `opentelemetry-api`; "
            "install with: pip install soweak[otel]"
        ) from e
    return trace


class OpenTelemetryAuditLog(AuditLog):
    """Bridge :class:`AuditEvent` to OpenTelemetry spans.

    Parameters:
      tracer: an OTEL ``Tracer`` (defaults to a tracer for the ``soweak``
        instrumentation library if not supplied).
      span_namespace: prefix for span names. Default ``"soweak"``.
      record_matched_text: include each signal's matched text on the span.
        Off by default — matched text often contains the sensitive value
        you're trying *not* to leak into telemetry.
    """

    def __init__(
        self,
        tracer: Any = None,
        span_namespace: str = "soweak",
        record_matched_text: bool = False,
    ) -> None:
        trace = _require_otel()
        self._tracer = tracer or trace.get_tracer("soweak")
        self._ns = span_namespace.rstrip(".")
        self._record_text = record_matched_text

    def record(self, event: AuditEvent) -> None:
        span_name = f"{self._ns}.{event.boundary.value}"
        with self._tracer.start_as_current_span(span_name) as span:
            span.set_attribute("soweak.request_id", event.request_id)
            span.set_attribute("soweak.boundary", event.boundary.value)
            span.set_attribute("soweak.action", event.decision.action.value)
            span.set_attribute("soweak.signal_count", len(event.signals))
            if event.decision.reason:
                span.set_attribute("soweak.decision.reason", event.decision.reason)
            for i, sig in enumerate(event.signals):
                attrs: dict[str, Any] = {
                    "soweak.detector": sig.detector,
                    "soweak.category": sig.category.value,
                    "soweak.severity": sig.severity.label,
                    "soweak.confidence": float(sig.confidence),
                }
                if sig.metadata.get("attack_type"):
                    attrs["soweak.attack_type"] = sig.metadata["attack_type"]
                if self._record_text and sig.matched_text:
                    attrs["soweak.matched_text"] = sig.matched_text
                span.add_event(name=f"signal[{i}]", attributes=attrs)


__all__ = ["OpenTelemetryAuditLog"]
