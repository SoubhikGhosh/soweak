"""Adapters wrap popular LLM clients/frameworks so soweak Pipelines run
transparently at the relevant boundary.

These are intentionally thin. Each adapter is opt-in via an extras install:

* ``pip install soweak[langchain]`` for :mod:`soweak.adapters.langchain`
* ``pip install soweak[openai]``    for :mod:`soweak.adapters.openai`
* ``pip install soweak[google]``    for :mod:`soweak.adapters.gemini`

Each adapter raises :class:`SecurityError` when the Pipeline blocks.
"""

from soweak.adapters.errors import SecurityError

__all__ = ["SecurityError"]
