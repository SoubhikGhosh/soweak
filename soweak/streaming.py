"""LLM10 streaming protections: repetition detection.

When an LLM gets stuck in a loop (the model emits the same sub-string over
and over), runtime costs balloon and the response is junk. The
:class:`RepetitionDetector` flags this from the streamed output so callers
can early-terminate.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

from soweak.core.detector import Detector, Signal
from soweak.core.types import Boundary, Context, OwaspCategory, Payload, Severity


class RepetitionDetector(Detector):
    """Flag output whose tail contains the same substring repeated ``min_repeats`` times.

    Defaults are tuned to catch model-loop pathology rather than legitimate
    repetition (e.g. a list of bullet points). Tune ``window_size`` for your
    expected output length and ``min_repeats`` for tolerance.
    """

    def __init__(
        self,
        window_size: int = 400,
        min_repeats: int = 5,
        unit_sizes: tuple[int, ...] = (3, 5, 10, 20, 40),
        boundaries: tuple[Boundary, ...] = (Boundary.OUTPUT, Boundary.STREAM),
        name: str = "repetition",
    ) -> None:
        if min_repeats < 2:
            raise ValueError("min_repeats must be >= 2")
        self._window = window_size
        self._min_repeats = min_repeats
        self._unit_sizes = unit_sizes
        self._boundaries = boundaries
        self._name = name
        # Pre-compile one regex per unit size.
        self._patterns: tuple[re.Pattern[str], ...] = tuple(
            re.compile(rf"(.{{{n},{n}}})\1{{{min_repeats - 1},}}", re.DOTALL)
            for n in unit_sizes
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> OwaspCategory:
        return OwaspCategory.LLM10_UNBOUNDED_CONSUMPTION

    @property
    def boundaries(self) -> tuple[Boundary, ...]:
        return self._boundaries

    def inspect(self, payload: Payload, ctx: Context) -> Iterable[Signal]:
        return self._iter(payload)

    def _iter(self, payload: Payload) -> Iterator[Signal]:
        text = payload.text
        if not text:
            return
        # Inspect only the tail so the regex stays cheap on long outputs.
        tail = text[-self._window :] if len(text) > self._window else text
        for n, pattern in zip(self._unit_sizes, self._patterns):
            match = pattern.search(tail)
            if match:
                unit = match.group(1)
                yield Signal(
                    detector=self._name,
                    category=OwaspCategory.LLM10_UNBOUNDED_CONSUMPTION,
                    severity=Severity.HIGH,
                    confidence=0.9,
                    message=(
                        f"Output repeats {self._min_repeats}+ times "
                        f"(unit={n} chars): {unit!r}"
                    ),
                    span=None,
                    matched_text=match.group(0)[:80],
                    metadata={
                        "unit_size": n,
                        "unit": unit,
                        "repeats": match.group(0).count(unit),
                    },
                )
                return  # one hit is enough


__all__ = ["RepetitionDetector"]
