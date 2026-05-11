"""PatternMatchDetector: a regex-driven Detector.

One instance is created per :class:`PatternPack`. Compilation happens once at
construction time; ``inspect`` is hot-path and only does ``finditer``.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

from soweak.core.detector import Detector, Signal
from soweak.core.types import Boundary, Context, OwaspCategory, Payload
from soweak.detectors.patterns import Pattern, PatternPack


class PatternMatchDetector(Detector):
    """Run a :class:`PatternPack` against payload text and emit signals."""

    def __init__(
        self,
        pack: PatternPack,
        boundaries: tuple[Boundary, ...] = (Boundary.INPUT,),
        name: str | None = None,
    ) -> None:
        self._pack = pack
        self._boundaries = boundaries
        self._name = name or f"pattern-match[{pack.name}]"
        self._compiled: tuple[tuple[re.Pattern[str], Pattern], ...] = tuple(
            (re.compile(p.regex, p.flags), p) for p in pack.patterns
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> OwaspCategory:
        return self._pack.category

    @property
    def boundaries(self) -> tuple[Boundary, ...]:
        return self._boundaries

    @property
    def pack(self) -> PatternPack:
        return self._pack

    def inspect(self, payload: Payload, ctx: Context) -> Iterable[Signal]:
        return self._iter(payload)

    def _iter(self, payload: Payload) -> Iterator[Signal]:
        text = payload.text
        if not text:
            return
        for regex, pat in self._compiled:
            for match in regex.finditer(text):
                yield Signal(
                    detector=self._name,
                    category=self._pack.category,
                    severity=pat.severity,
                    confidence=pat.confidence,
                    message=pat.description,
                    span=match.span(),
                    matched_text=match.group(0),
                    metadata={
                        "attack_type": pat.attack_type,
                        "pattern": pat.regex,
                    },
                )
