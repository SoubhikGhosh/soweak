"""OpenAI adapter.

Install with ``pip install soweak[openai]``.

Wraps an OpenAI client so every ``chat.completions.create`` call has its
inputs scanned at the input boundary and (when not streamed) its output
scanned at the output boundary, using a soweak ``Pipeline``.
"""

from __future__ import annotations

from typing import Any

from soweak import Context, Pipeline
from soweak.adapters.errors import SecurityError

try:  # pragma: no cover - optional dep
    from openai import OpenAI  # noqa: F401  (presence check only)
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "openai integration requires `openai>=1.0`; "
        "install with: pip install soweak[openai]"
    ) from e


class SecureOpenAI:
    """Wraps an OpenAI client and applies a soweak Pipeline at I/O boundaries.

    Example::

        from openai import OpenAI
        from soweak.adapters.openai import SecureOpenAI

        client = SecureOpenAI(OpenAI(), pipeline=my_pipeline)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "..."}],
        )
    """

    def __init__(self, client: Any, pipeline: Pipeline) -> None:
        self._client = client
        self._pipeline = pipeline
        self.chat = _Chat(client.chat, pipeline)


class _Chat:
    def __init__(self, chat: Any, pipeline: Pipeline) -> None:
        self._chat = chat
        self._pipeline = pipeline
        self.completions = _Completions(chat.completions, pipeline)


class _Completions:
    def __init__(self, completions: Any, pipeline: Pipeline) -> None:
        self._completions = completions
        self._pipeline = pipeline

    def create(self, *, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        ctx = Context()
        # Input boundary: scan every user/tool/assistant content chunk.
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                decision = self._pipeline.check_input(content, ctx)
                if decision.blocked:
                    raise SecurityError(decision)
        response = self._completions.create(messages=messages, **kwargs)

        if kwargs.get("stream"):
            return response  # streaming output scan is Phase 2

        # Output boundary: scan each choice's content.
        for choice in getattr(response, "choices", []):
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                decision = self._pipeline.check_output(content, ctx)
                if decision.blocked:
                    raise SecurityError(decision)
        return response


__all__ = ["SecureOpenAI"]
