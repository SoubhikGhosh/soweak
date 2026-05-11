"""LangChain adapter.

Install with ``pip install soweak[langchain]``.

Exposes a ``Runnable`` that runs a soweak Pipeline at a chosen boundary, plus a
``BaseCallbackHandler`` that hooks the same Pipeline into LLM start/end events.
"""

from __future__ import annotations

from typing import Any

from soweak import Context, Pipeline
from soweak.adapters.errors import SecurityError
from soweak.core.types import Boundary

try:  # pragma: no cover - optional dep
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.runnables import RunnableLambda
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "langchain integration requires `langchain-core`; "
        "install with: pip install soweak[langchain]"
    ) from e


def guard_runnable(
    pipeline: Pipeline,
    boundary: Boundary = Boundary.INPUT,
    on_block: str = "raise",
) -> RunnableLambda:
    """Return a ``RunnableLambda`` that runs ``pipeline`` against each item.

    ``on_block`` controls behaviour when the pipeline blocks:

    * ``"raise"``   (default): raise :class:`SecurityError`
    * ``"redact"``  : pass the transformed payload through (no error)
    * ``"drop"``    : return ``None``
    """

    def _check(payload_text: str) -> str | None:
        if boundary == Boundary.INPUT:
            decision = pipeline.check_input(payload_text)
        elif boundary == Boundary.OUTPUT:
            decision = pipeline.check_output(payload_text)
        else:
            raise ValueError(f"unsupported boundary for runnable: {boundary}")
        if decision.blocked:
            if on_block == "raise":
                raise SecurityError(decision)
            if on_block == "drop":
                return None
        return decision.payload.text

    return RunnableLambda(_check)


class SoweakCallbackHandler(BaseCallbackHandler):
    """Run a Pipeline on each LLM start (input) and end (output)."""

    def __init__(self, pipeline: Pipeline, raise_on_block: bool = True) -> None:
        self._pipeline = pipeline
        self._raise = raise_on_block

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        ctx = Context()
        for prompt in prompts:
            decision = self._pipeline.check_input(prompt, ctx)
            if decision.blocked and self._raise:
                raise SecurityError(decision)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        ctx = Context()
        for generation_group in getattr(response, "generations", []):
            for gen in generation_group:
                text = getattr(gen, "text", None) or ""
                decision = self._pipeline.check_output(text, ctx)
                if decision.blocked and self._raise:
                    raise SecurityError(decision)


__all__ = ["SoweakCallbackHandler", "guard_runnable"]
