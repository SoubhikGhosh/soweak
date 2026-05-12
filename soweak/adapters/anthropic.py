"""Anthropic adapter.

Install with ``pip install soweak[anthropic]``.

Wraps an Anthropic client so every ``messages.create`` call has its inputs
scanned at the input boundary and (when not streamed) its outputs scanned
at the output boundary, using a soweak ``Pipeline``.
"""

from __future__ import annotations

from typing import Any

from soweak import Context, Pipeline
from soweak.adapters.errors import SecurityError

try:  # pragma: no cover - optional dep
    import anthropic  # noqa: F401  (presence check only)
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "anthropic integration requires `anthropic>=0.34`; "
        "install with: pip install soweak[anthropic]"
    ) from e


class SecureAnthropic:
    """Wraps an Anthropic client and applies a soweak Pipeline at I/O boundaries.

    Example::

        from anthropic import Anthropic
        from soweak.adapters.anthropic import SecureAnthropic

        client = SecureAnthropic(Anthropic(), pipeline=my_pipeline)
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "..."}],
        )
    """

    def __init__(self, client: Any, pipeline: Pipeline) -> None:
        self._client = client
        self._pipeline = pipeline
        self.messages = _Messages(client.messages, pipeline)


class _Messages:
    def __init__(self, messages: Any, pipeline: Pipeline) -> None:
        self._messages = messages
        self._pipeline = pipeline

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        ctx = Context()
        # Input boundary: scan every user/assistant message's text content.
        for msg in messages:
            content = msg.get("content")
            for chunk in _iter_message_text(content):
                decision = self._pipeline.check_input(chunk, ctx)
                if decision.blocked:
                    raise SecurityError(decision)

        response = self._messages.create(
            messages=messages, system=system, **kwargs
        )

        if kwargs.get("stream"):
            return response  # streaming output scan handled by StreamingPipeline

        # Output boundary: scan each text block on the response.
        content = getattr(response, "content", None) or []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                decision = self._pipeline.check_output(text, ctx)
                if decision.blocked:
                    raise SecurityError(decision)
        return response


def _iter_message_text(content: Any) -> list[str]:
    """Anthropic content can be a string or a list of content blocks."""
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        out.append(text)
            elif isinstance(block, str):
                out.append(block)
        return out
    return []


__all__ = ["SecureAnthropic"]
