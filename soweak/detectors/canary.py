"""CanaryDetector: detect system-prompt leakage on the output boundary.

Place unique canary tokens inside your system prompt (or any context the model
should not echo verbatim). Scan model output for those tokens — any hit
indicates the model is regurgitating privileged context.

Recommended use::

    CANARIES = ["x7K2-PRODSEC-9F4E", "internal-arch-doc-2026"]

    system_prompt = (
        f"# canary: {CANARIES[0]}\\n"
        "You are an assistant for ACME Inc..."
    )

    pipeline = Pipeline(
        Policy.builder()
        .on_output()
            .detect(CanaryDetector(tokens=CANARIES))
            .enforce(BlockEnforcer(min_severity=Severity.CRITICAL))
        .build()
    )
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

from soweak.core.detector import Detector, Signal
from soweak.core.types import Boundary, Context, OwaspCategory, Payload, Severity


class CanaryDetector(Detector):
    """Detect canary tokens appearing in payload text (typically model output)."""

    def __init__(
        self,
        tokens: Iterable[str],
        boundaries: tuple[Boundary, ...] = (Boundary.OUTPUT, Boundary.STREAM),
        name: str = "canary",
        severity: Severity = Severity.CRITICAL,
    ) -> None:
        toks = [t for t in tokens if t]
        if not toks:
            raise ValueError("CanaryDetector requires at least one non-empty token")
        self._tokens = tuple(toks)
        self._boundaries = boundaries
        self._name = name
        self._severity = severity
        self._regex = re.compile(
            "|".join(re.escape(t) for t in self._tokens)
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> OwaspCategory:
        return OwaspCategory.LLM07_SYSTEM_PROMPT_LEAKAGE

    @property
    def boundaries(self) -> tuple[Boundary, ...]:
        return self._boundaries

    @property
    def tokens(self) -> tuple[str, ...]:
        return self._tokens

    def inspect(self, payload: Payload, ctx: Context) -> Iterable[Signal]:
        return self._iter(payload)

    def _iter(self, payload: Payload) -> Iterator[Signal]:
        text = payload.text
        if not text:
            return
        for match in self._regex.finditer(text):
            yield Signal(
                detector=self._name,
                category=OwaspCategory.LLM07_SYSTEM_PROMPT_LEAKAGE,
                severity=self._severity,
                confidence=1.0,
                message=f"Canary token leaked in output: {match.group(0)!r}",
                span=match.span(),
                matched_text=match.group(0),
                metadata={"attack_type": "system_prompt_leak"},
            )
