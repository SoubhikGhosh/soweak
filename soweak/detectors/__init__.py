"""Built-in detectors and pattern packs.

The framework is detector-agnostic — anything implementing
:class:`soweak.core.Detector` works. This module ships the baseline detectors
that v3.0 covers: a configurable :class:`PatternMatchDetector` and a
:class:`CanaryDetector` for output-side system-prompt leakage.
"""

from soweak.detectors.canary import CanaryDetector
from soweak.detectors.pattern_match import PatternMatchDetector
from soweak.detectors.patterns import (
    INPUT_DLP_PACK,
    PROMPT_INJECTION_PACK,
    SYSTEM_PROMPT_EXTRACTION_PACK,
    Pattern,
    PatternPack,
)


def prompt_injection_detector() -> PatternMatchDetector:
    """Default LLM01 (prompt injection) detector for the input boundary."""
    return PatternMatchDetector(PROMPT_INJECTION_PACK)


def input_dlp_detector() -> PatternMatchDetector:
    """Default LLM02 input DLP detector (PII / secrets / credentials)."""
    return PatternMatchDetector(INPUT_DLP_PACK)


def system_prompt_extraction_detector() -> PatternMatchDetector:
    """Default LLM07 detector for system-prompt extraction attempts on input."""
    return PatternMatchDetector(SYSTEM_PROMPT_EXTRACTION_PACK)


__all__ = [
    "CanaryDetector",
    "INPUT_DLP_PACK",
    "PROMPT_INJECTION_PACK",
    "Pattern",
    "PatternMatchDetector",
    "PatternPack",
    "SYSTEM_PROMPT_EXTRACTION_PACK",
    "input_dlp_detector",
    "prompt_injection_detector",
    "system_prompt_extraction_detector",
]
