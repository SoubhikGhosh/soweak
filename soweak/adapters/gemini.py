"""Google Gemini adapter.

Install with ``pip install soweak[google]``.

Wraps a ``google.generativeai`` ``GenerativeModel`` so every ``generate_content``
call has its inputs scanned at the input boundary and its outputs at the
output boundary, using a soweak ``Pipeline``.
"""

from __future__ import annotations

from typing import Any, Iterable

from soweak import Context, Pipeline
from soweak.adapters.errors import SecurityError

try:  # pragma: no cover - optional dep
    import google.generativeai as genai  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "gemini integration requires `google-generativeai>=0.3`; "
        "install with: pip install soweak[google]"
    ) from e


class SecureGemini:
    """Wrap a Gemini ``GenerativeModel`` with a soweak Pipeline."""

    def __init__(self, model: Any, pipeline: Pipeline) -> None:
        self._model = model
        self._pipeline = pipeline

    def generate_content(
        self,
        contents: str | Iterable[Any],
        **kwargs: Any,
    ) -> Any:
        ctx = Context()
        for text in _iter_text(contents):
            decision = self._pipeline.check_input(text, ctx)
            if decision.blocked:
                raise SecurityError(decision)

        response = self._model.generate_content(contents, **kwargs)

        output_text = getattr(response, "text", None)
        if isinstance(output_text, str):
            decision = self._pipeline.check_output(output_text, ctx)
            if decision.blocked:
                raise SecurityError(decision)
        return response

    def start_chat(self, **kwargs: Any) -> SecureChat:
        chat = self._model.start_chat(**kwargs)
        return SecureChat(chat, self._pipeline)


class SecureChat:
    def __init__(self, chat: Any, pipeline: Pipeline) -> None:
        self._chat = chat
        self._pipeline = pipeline

    def send_message(self, message: str, **kwargs: Any) -> Any:
        ctx = Context()
        decision = self._pipeline.check_input(message, ctx)
        if decision.blocked:
            raise SecurityError(decision)
        response = self._chat.send_message(message, **kwargs)
        output = getattr(response, "text", None)
        if isinstance(output, str):
            decision = self._pipeline.check_output(output, ctx)
            if decision.blocked:
                raise SecurityError(decision)
        return response


def _iter_text(contents: str | Iterable[Any]) -> Iterable[str]:
    if isinstance(contents, str):
        yield contents
        return
    for c in contents:
        if isinstance(c, str):
            yield c
        elif isinstance(c, dict):
            text = c.get("text") or c.get("content")
            if isinstance(text, str):
                yield text


__all__ = ["SecureChat", "SecureGemini"]
